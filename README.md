# 🛰️ Deepfake Intelligence Radar — Image Triage

A fast, visual triage tool that helps you assess image authenticity. It blends classic Error Level Analysis (ELA), frequency-domain analysis (FFT), and face landmark detection to produce an overall suspicion score for each image. Ideal for hackathons, newsroom triage, and rapid investigations.

## ✨ Purpose
- Quickly flag potentially manipulated, AI-generated, or suspicious images
- Provide both numerical indicators and visual heatmaps (ELA) to aid judgement
- Batch process many images and export results as CSV

## 🧰 Tech Stack
- UI: [Streamlit](https://streamlit.io/) (interactive, wide layout)
- DataFrame engine: [Daft (getdaft)](https://www.getdaft.io/) for scalable UDF pipelines
- CV/Imaging: OpenCV (headless), Pillow, NumPy
- Face detection: MediaPipe FaceMesh
- Dev environment: Daytona + Dev Containers (VS Code friendly)

## ⚡ Quickstart

### Prerequisites
- Python 3.10+
- pip
- (Windows) PowerShell; (Linux/Mac) Bash

### 1) Clone and enter the project
```bash
# using bash
# git clone <your-fork-or-repo-url>
# cd diradar
```

### 2) Start via bootstrap
- Linux/Mac/WSL
```bash
chmod +x bootstrap.sh
./bootstrap.sh
```
- Windows (PowerShell)
```powershell
# If needed on first run
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Run
./bootstrap.ps1
```

The app will start on:
- http://localhost:7860

### 3) Use the App
- Upload one or more JPG/PNG images
- Click "Analyze"
- Review the results table, ELA heatmaps, and download CSV

## 🧪 Sample Data
Generate sample test images locally (optional):
```bash
python create_test_images.py
# Then upload files from the test_images/ folder in the app
```

## ☁️ Daytona Sandbox (Optional)
This repo includes configuration files to accelerate cloud dev environments.

- Dev Container / Daytona config: `.devcontainer.json`, `.daytona.yaml` or `daytona.yaml`
- Script to initialize Daytona workspace (manual API-key step required): `init-daytona.sh`

Example flow (bash):
```bash
# Install daytona CLI if needed
curl -L https://download.daytona.io/daytona/install.sh | bash

# Authenticate separately (do not hardcode keys)
daytona auth login --api-key <YOUR_DAYTONA_API_KEY>

# Create a workspace from this repo
# From the project root:
daytona create .

# In the workspace terminal, run the app
./bootstrap.sh
```

## 🐳 Docker (Optional)
This repo includes a Dockerfile and docker-compose for quick container runs.

```bash
# Build and run with Docker Compose
docker-compose up --build
# App at http://localhost:7860

# OR plain Docker
docker build -t diradar .
docker run -p 7860:7860 diradar
```

## 🔬 How It Works
- ELA (Error Level Analysis): recompress JPEG and compute pixel-wise difference; visualize differences and compute mean-intensity score
- FFT Frequency Analysis: grayscale + resize → 2D FFT → measure high-frequency energy ratio
- Face Detection: MediaPipe FaceMesh returns a binary signal if a face is present
- Suspicion Score: a blended metric of ELA + FFT + face presence with context-aware boosts

See implementation in `pipeline.py`:
- `ela_bytes(image_bytes) -> (png_heatmap_bytes, ela_score)`
- `fft_score(image_bytes) -> float`
- `face_landmark_conf(image_bytes) -> float`
- `suspicion(image_bytes) -> float`
- `build_df(paths) -> Daft DataFrame`

## 🖼️ UI Overview & Sample Screenshots
Add your screenshots to the paths below after running the app (press Analyze and take screenshots):

- Results table
  - `docs/screenshots/results_table.png`
  - Markdown embed:
    ```
    ![Results Table](docs/screenshots/results_table.png)
    ```

- ELA heatmap grid
  - `docs/screenshots/heatmap_grid.png`
  - Markdown embed:
    ```
    ![ELA Heatmap Grid](docs/screenshots/heatmap_grid.png)
    ```

Tip: On macOS use Shift+Cmd+4, on Windows use Snipping Tool, on Linux use your screenshot utility. Save to `docs/screenshots/` and commit.

## 📤 Exporting Results
After analysis, click "Download Results as CSV" to export per-image scores. Useful for sharing, downstream dashboards, or investigations.

## 🧩 Project Structure
```
.
├─ app.py                 # Streamlit UI
├─ pipeline.py            # ELA, FFT, Face, Suspicion, Daft pipeline
├─ requirements.txt       # Pinned main deps
├─ requirements-full.txt  # Full lock snapshot (optional)
├─ bootstrap.sh           # Bash setup + run
├─ bootstrap.ps1          # PowerShell setup + run (Windows)
├─ Dockerfile             # Container build
├─ docker-compose.yml     # Orchestration
├─ daytona.yaml / .devcontainer.json  # Cloud/dev env
├─ create_test_images.py  # Optional sample image generator
└─ docs/
   └─ screenshots/        # Place UI screenshots here
```

## 🛠️ Troubleshooting
- If port 7860 is in use, stop the existing process or change the port in the run command
- If MediaPipe fails to load on your platform, ensure system deps are installed (see Dockerfile for hints)
- On Windows, run PowerShell as Administrator when needed for dependency installs

## 🔒 Security Notes
- Never commit or paste API keys in code or README
- Configure secrets via environment variables or CLIs (`daytona auth login`)

## 📄 License
MIT (or your preferred license)

