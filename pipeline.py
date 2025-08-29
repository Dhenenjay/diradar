from __future__ import annotations

from io import BytesIO
from typing import Tuple, Optional, List, Union
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageEnhance, ImageOps
import cv2
import mediapipe as mp
import daft
from daft import col, DataType


def _cv2_colormap(name: str) -> int:
    """Map a human-friendly colormap name to an OpenCV colormap constant."""
    name = (name or "").strip().lower()
    mapping = {
        "inferno": cv2.COLORMAP_INFERNO,
        "jet": cv2.COLORMAP_JET,
        "magma": cv2.COLORMAP_MAGMA,
        "plasma": cv2.COLORMAP_PLASMA,
        "turbo": cv2.COLORMAP_TURBO,
        "hot": cv2.COLORMAP_HOT,
    }
    return mapping.get(name, cv2.COLORMAP_INFERNO)


def ela_bytes(
    image_bytes: bytes,
    quality: int = 90,
    colormap: str = "inferno",
    scale: Optional[float] = None,
) -> Tuple[bytes, float]:
    """
    Perform Error Level Analysis (ELA) on an image provided as bytes.

    The function recompresses the image as JPEG at the given quality, computes the
    pixel-wise difference to the original, amplifies the differences for visibility,
    applies a heatmap, and returns the heatmap as PNG bytes along with a float score.

    Parameters
    - image_bytes: Raw bytes of the input image.
    - quality: JPEG quality (1-100) used for recompression during ELA. Typical values are 85-95.
    - colormap: Name of OpenCV colormap to apply (inferno, jet, magma, plasma, turbo, hot).
    - scale: Optional manual amplification factor for the difference visualization.
             If None, an automatic scale is chosen so the max difference maps near 255,
             with a safety cap.

    Returns
    - (png_heatmap_bytes, score):
        - png_heatmap_bytes: Bytes of the ELA heatmap image encoded as PNG.
        - score: Float in [0, 1], representing the average (unscaled) difference intensity
                 normalized to 0..1. Higher means more compression-inconsistent areas.

    Raises
    - ValueError: If the input image bytes cannot be decoded.
    """
    if not isinstance(image_bytes, (bytes, bytearray, memoryview)):
        raise ValueError("image_bytes must be a bytes-like object")

    if not (1 <= int(quality) <= 100):
        raise ValueError("quality must be in the range [1, 100]")

    # Decode input image
    try:
        original = Image.open(BytesIO(image_bytes))
        # Apply EXIF orientation if present and ensure RGB
        original = ImageOps.exif_transpose(original).convert("RGB")
    except Exception as e:
        raise ValueError("Failed to decode image bytes") from e

    # Recompress as JPEG at the specified quality
    jpeg_buf = BytesIO()
    original.save(jpeg_buf, format="JPEG", quality=int(quality))
    jpeg_buf.seek(0)
    recompressed = Image.open(jpeg_buf).convert("RGB")

    # Compute difference (unscaled)
    diff = ImageChops.difference(original, recompressed)

    # Compute a score from the unscaled grayscale difference (independent of visualization scaling)
    diff_gray_unscaled = diff.convert("L")
    arr_unscaled = np.asarray(diff_gray_unscaled, dtype=np.float32)
    score = float(arr_unscaled.mean() / 255.0)

    # Determine visualization scale
    if scale is None:
        # Auto-scale so that the maximum difference maps near 255; cap to avoid over-amplification
        extrema = diff.getextrema()  # [(min, max) per channel]
        max_diff = max(ch_max for (_ch_min, ch_max) in extrema) if isinstance(extrema, (list, tuple)) else 0
        auto_scale = (255.0 / max_diff) if max_diff else 1.0
        s = float(min(auto_scale, 15.0))  # cap
    else:
        s = float(scale)
        if s <= 0:
            s = 1.0

    # Enhance brightness (linear amplification of differences)
    diff_enhanced = ImageEnhance.Brightness(diff).enhance(s)

    # Convert to grayscale for heatmap mapping
    gray = diff_enhanced.convert("L")  # uint8
    gray_arr = np.asarray(gray)

    # Apply heatmap colormap using OpenCV (expects uint8 single channel)
    cmap_const = _cv2_colormap(colormap)
    heatmap_bgr = cv2.applyColorMap(gray_arr, cmap_const)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    # Encode to PNG bytes
    out_img = Image.fromarray(heatmap_rgb)
    out_buf = BytesIO()
    out_img.save(out_buf, format="PNG")
    png_bytes = out_buf.getvalue()

    return png_bytes, score


def fft_score(
    image_bytes: bytes,
    target_size: int = 256,
    high_band: tuple[float, float] = (0.35, 0.95),
    apply_window: bool = True,
) -> float:
    """
    Compute a frequency-artifact score from an image using the 2D FFT.

    The image is converted to grayscale, resized to a square, optionally windowed
    (Hann) to reduce edge effects, transformed via FFT, and the score is computed
    as the ratio of high-frequency energy to total energy within a specified
    normalized radial band of the spectrum.

    Parameters
    - image_bytes: Raw bytes of the input image.
    - target_size: The size (pixels) to which the image is resized (target_size x target_size).
    - high_band: Tuple (low, high) specifying the normalized radius band [0..1]
                 considered "high-frequency" for scoring. Defaults to (0.35, 0.95).
    - apply_window: Whether to apply a 2D Hann window before FFT to reduce spectral leakage.

    Returns
    - score: Float in [0, 1], representing the proportion of energy in the specified
             high-frequency band. Higher values indicate more high-frequency content,
             which may correlate with compression artifacts or noise.

    Raises
    - ValueError: If the input image bytes cannot be decoded or parameters are invalid.
    """
    if not isinstance(image_bytes, (bytes, bytearray, memoryview)):
        raise ValueError("image_bytes must be a bytes-like object")

    if int(target_size) < 32:
        raise ValueError("target_size must be >= 32")

    try:
        img = Image.open(BytesIO(image_bytes))
        img = ImageOps.exif_transpose(img).convert("L")  # grayscale
    except Exception as e:
        raise ValueError("Failed to decode image bytes") from e

    # Resize to a square of target_size for a stable FFT grid
    size = int(target_size)
    img = img.resize((size, size), resample=Image.LANCZOS)

    arr = np.asarray(img, dtype=np.float32) / 255.0
    # Remove DC bias to avoid over-dominating the spectrum center
    arr = arr - float(arr.mean())

    if apply_window:
        wy = np.hanning(size).astype(np.float32)
        wx = np.hanning(size).astype(np.float32)
        window2d = np.outer(wy, wx)
        arr = arr * window2d

    # 2D FFT and power spectrum
    F = np.fft.fft2(arr)
    S = np.fft.fftshift(F)
    power = np.abs(S) ** 2  # power spectrum

    # Build normalized radial distance map from the center
    h, w = power.shape
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    yy = np.arange(h, dtype=np.float32)[:, None]
    xx = np.arange(w, dtype=np.float32)[None, :]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r_norm = r / (r.max() if r.max() > 0 else 1.0)

    low, high = float(high_band[0]), float(high_band[1])
    # Sanitize band bounds
    if not (0.0 <= low < high <= 1.0):
        low, high = 0.35, 0.95

    hi_mask = (r_norm >= low) & (r_norm <= high)

    total_energy = float(power.sum())
    if total_energy <= 1e-12 or not np.isfinite(total_energy):
        return 0.0

    high_energy = float(power[hi_mask].sum())
    score = high_energy / total_energy

    # Clamp to [0,1] and ensure a plain float is returned
    if not np.isfinite(score):
        score = 0.0
    score = max(0.0, min(1.0, float(score)))
    return score


def face_landmark_conf(
    image_bytes: bytes,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
    refine_landmarks: bool = True,
) -> float:
    """
    Detect faces using MediaPipe FaceMesh and return binary confidence.

    Uses MediaPipe's FaceMesh model to detect facial landmarks. Returns 1.0 if
    at least one face is detected with sufficient confidence, 0.0 otherwise.

    Parameters
    - image_bytes: Raw bytes of the input image.
    - min_detection_confidence: Minimum confidence for face detection (0.0-1.0).
    - min_tracking_confidence: Minimum confidence for landmark tracking (0.0-1.0).
    - refine_landmarks: Whether to refine face landmarks around lips and eyes.

    Returns
    - confidence: 1.0 if face detected, 0.0 otherwise.

    Raises
    - ValueError: If the input image bytes cannot be decoded.
    
    Notes
    - FaceMesh detects up to 468 3D facial landmarks.
    - The function returns a binary result but could be extended to return
      continuous confidence scores or landmark positions if needed.
    """
    if not isinstance(image_bytes, (bytes, bytearray, memoryview)):
        raise ValueError("image_bytes must be a bytes-like object")

    # Validate confidence parameters
    min_detection_confidence = max(0.0, min(1.0, float(min_detection_confidence)))
    min_tracking_confidence = max(0.0, min(1.0, float(min_tracking_confidence)))

    try:
        # Load image from bytes
        img = Image.open(BytesIO(image_bytes))
        img = ImageOps.exif_transpose(img).convert("RGB")
        
        # Convert PIL image to numpy array for MediaPipe
        img_array = np.asarray(img, dtype=np.uint8)
        
    except Exception as e:
        raise ValueError("Failed to decode image bytes") from e

    # Initialize MediaPipe FaceMesh
    mp_face_mesh = mp.solutions.face_mesh
    
    try:
        # Use FaceMesh in static image mode
        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,  # Only check for at least one face
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        ) as face_mesh:
            
            # Process the image
            results = face_mesh.process(img_array)
            
            # Check if any face landmarks were detected
            if results.multi_face_landmarks and len(results.multi_face_landmarks) > 0:
                # At least one face detected
                return 1.0
            else:
                # No faces detected
                return 0.0
                
    except Exception as e:
        # If processing fails, return no detection
        # This could happen with corrupted images or other issues
        return 0.0


def suspicion(
    image_bytes: bytes,
    ela_weight: float = 0.35,
    fft_weight: float = 0.35,
    face_weight: float = 0.30,
    face_boost_factor: float = 1.5,
    ela_quality: int = 90,
) -> float:
    """
    Calculate a comprehensive suspicion score by blending multiple analysis techniques.

    Combines Error Level Analysis (ELA), Fast Fourier Transform (FFT) frequency analysis,
    and face detection confidence to produce a final suspicion score. The score indicates
    the likelihood that an image has been manipulated, is AI-generated, or is otherwise
    suspicious.

    Parameters
    - image_bytes: Raw bytes of the input image.
    - ela_weight: Weight for ELA score contribution (default: 0.35).
    - fft_weight: Weight for FFT score contribution (default: 0.35).
    - face_weight: Weight for face detection impact (default: 0.30).
    - face_boost_factor: Multiplier applied when face is detected (default: 1.5).
                        Higher values increase suspicion for face-containing images.
    - ela_quality: JPEG quality parameter for ELA analysis (default: 90).

    Returns
    - score: Float in [0, 1], where:
            0.0-0.3: Low suspicion (likely authentic)
            0.3-0.6: Moderate suspicion (possible manipulation)
            0.6-0.8: High suspicion (likely manipulated)
            0.8-1.0: Very high suspicion (strong evidence of manipulation)

    Raises
    - ValueError: If weights don't sum to 1.0 or image bytes are invalid.

    Notes
    - The function is optimized for detecting:
      * AI-generated images (especially faces)
      * Edited/manipulated photos
      * Deepfakes and face swaps
      * Heavy compression artifacts
    - Face detection adds context: manipulated faces are often more concerning.
    """
    if not isinstance(image_bytes, (bytes, bytearray, memoryview)):
        raise ValueError("image_bytes must be a bytes-like object")

    # Normalize weights to ensure they sum to 1.0
    total_weight = ela_weight + fft_weight + face_weight
    if abs(total_weight - 1.0) > 0.001:  # Allow small floating point errors
        # Auto-normalize if weights don't sum to 1
        ela_weight = ela_weight / total_weight
        fft_weight = fft_weight / total_weight
        face_weight = face_weight / total_weight

    try:
        # Run ELA analysis
        _, ela_score = ela_bytes(
            image_bytes,
            quality=ela_quality,
            colormap="inferno",
        )
    except Exception:
        # If ELA fails, use neutral score
        ela_score = 0.5

    try:
        # Run FFT analysis
        fft_result = fft_score(
            image_bytes,
            target_size=256,
            high_band=(0.35, 0.95),
            apply_window=True,
        )
    except Exception:
        # If FFT fails, use neutral score
        fft_result = 0.5

    try:
        # Check for face presence
        face_detected = face_landmark_conf(
            image_bytes,
            min_detection_confidence=0.5,
            refine_landmarks=True,
        )
    except Exception:
        # If face detection fails, assume no face
        face_detected = 0.0

    # Apply non-linear transformations to enhance signal
    
    # ELA: Higher scores indicate more compression artifacts/edits
    # Apply slight exponential scaling to emphasize higher values
    ela_contribution = ela_score ** 0.85
    
    # FFT: High frequency content suggests artifacts
    # Apply sigmoid-like transformation to create better separation
    fft_contribution = 1.0 / (1.0 + np.exp(-10 * (fft_result - 0.5)))
    
    # Face detection: Presence of face increases concern for deepfakes
    if face_detected > 0.5:
        # Face detected - apply boost factor and use face confidence
        face_contribution = face_weight * face_boost_factor
        # Also slightly boost ELA and FFT scores for face images
        ela_contribution = min(1.0, ela_contribution * 1.1)
        fft_contribution = min(1.0, fft_contribution * 1.1)
    else:
        # No face - reduce the face component's influence
        face_contribution = face_weight * 0.2
    
    # Weighted combination
    base_score = (
        ela_weight * ela_contribution +
        fft_weight * fft_contribution +
        face_contribution
    )
    
    # Apply context-aware adjustments
    
    # If both ELA and FFT are high, likely manipulation
    if ela_contribution > 0.7 and fft_contribution > 0.7:
        base_score = min(1.0, base_score * 1.2)
    
    # If face detected with high ELA, very suspicious (possible deepfake)
    if face_detected > 0.5 and ela_contribution > 0.6:
        base_score = min(1.0, base_score * 1.15)
    
    # If very low ELA and FFT, reduce suspicion
    if ela_contribution < 0.2 and fft_contribution < 0.2:
        base_score = base_score * 0.7
    
    # Final normalization and clamping
    final_score = max(0.0, min(1.0, base_score))
    
    # Apply subtle smoothing to avoid extreme edge cases
    if final_score < 0.05:
        final_score = final_score * 2  # Very low scores stay very low
    elif final_score > 0.95:
        final_score = 0.95 + (final_score - 0.95) * 0.5  # Cap extreme highs
    
    return float(final_score)


# ---------- Daft DataFrame pipeline ----------

def _read_file_bytes(path: Union[str, Path]) -> bytes:
    p = Path(path)
    with p.open("rb") as f:
        return f.read()


def build_df(paths: Union[List[Union[str, Path]], Union[str, Path]]) -> daft.DataFrame:
    """
    Build a Daft DataFrame from local file paths and compute analysis columns.

    Steps:
    - Normalize and validate input paths
    - Load bytes for each file
    - Apply UDFs: ela_bytes, fft_score, face_landmark_conf
    - Extract ELA score and ELA PNG bytes from the ela_bytes output
    - Compute final suspicion score

    Columns produced:
    - path: string path
    - bytes: binary image bytes
    - ela_png: PNG bytes of ELA heatmap
    - ela_score: float32
    - fft_score: float32
    - face_conf: float32 (0.0 or 1.0)
    - suspicion: float32 in [0, 1]
    """
    # Normalize input to a list
    if isinstance(paths, (str, Path)):
        paths = [paths]
    paths = [str(Path(p)) for p in paths]

    # Build initial Daft DataFrame
    df = daft.from_pydict({"path": paths})

    # Read file bytes as a new column
    df = df.with_column("bytes", col("path").apply(_read_file_bytes, return_dtype=DataType.binary()))

    # Apply ELA UDF -> returns (png_bytes, score)
    df = df.with_column(
        "ela_tuple",
        col("bytes").apply(
            lambda b: ela_bytes(b, quality=90, colormap="inferno"),
            return_dtype=DataType.python(),
        ),
    )

    # Extract ela_png and ela_score from the tuple
    df = df.with_columns(
        {
            "ela_png": col("ela_tuple").apply(lambda t: t[0], return_dtype=DataType.binary()),
            "ela_score": col("ela_tuple").apply(lambda t: float(t[1]), return_dtype=DataType.float32()),
        }
    )
    # Drop the ela_tuple column by selecting all others
    df = df.select(col("path"), col("bytes"), col("ela_png"), col("ela_score"))

    # FFT score
    df = df.with_column(
        "fft_score",
        col("bytes").apply(lambda b: float(fft_score(b)), return_dtype=DataType.float32()),
    )

    # Face confidence
    df = df.with_column(
        "face_conf",
        col("bytes").apply(lambda b: float(face_landmark_conf(b)), return_dtype=DataType.float32()),
    )

    # Suspicion score - compute directly from bytes
    df = df.with_column(
        "suspicion",
        col("bytes").apply(
            lambda b: float(suspicion(b, ela_weight=0.35, fft_weight=0.35, face_weight=0.30, face_boost_factor=1.5, ela_quality=90)),
            return_dtype=DataType.float32(),
        ),
    )

    return df
