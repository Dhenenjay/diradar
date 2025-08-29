"""
Create sample test images for verifying the Deepfake Intelligence Radar app
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

# Create test_images directory
os.makedirs("test_images", exist_ok=True)

# 1. Create a simple authentic-looking image (low suspicion expected)
print("Creating authentic-looking test image...")
img1 = Image.new('RGB', (512, 512), color='white')
draw = ImageDraw.Draw(img1)

# Add gradient-like pattern
for i in range(0, 256, 10):
    color = (i, 100 + i // 2, 200 - i // 2)
    draw.rectangle([i, i, 512-i, 512-i], outline=color, width=2)

# Add some natural noise
pixels = np.array(img1)
noise = np.random.normal(0, 5, pixels.shape)
noisy = np.clip(pixels + noise, 0, 255).astype(np.uint8)
img1 = Image.fromarray(noisy)
img1.save("test_images/authentic_gradient.jpg", quality=95)
print("✓ Saved: authentic_gradient.jpg")

# 2. Create a heavily compressed image (high ELA score expected)
print("Creating heavily compressed test image...")
img2 = Image.new('RGB', (512, 512), color='blue')
draw2 = ImageDraw.Draw(img2)

# Add shapes with different colors
draw2.rectangle([50, 50, 200, 200], fill='red')
draw2.ellipse([250, 250, 450, 450], fill='green')
draw2.polygon([(300, 50), (450, 150), (250, 150)], fill='yellow')

# Save with low quality to introduce compression artifacts
img2.save("test_images/compressed_shapes.jpg", quality=20)
# Reload and save again to compound compression
img2_reload = Image.open("test_images/compressed_shapes.jpg")
img2_reload.save("test_images/compressed_shapes.jpg", quality=30)
print("✓ Saved: compressed_shapes.jpg")

# 3. Create an image with artificial patterns (high FFT score expected)
print("Creating image with artificial patterns...")
img3 = Image.new('RGB', (512, 512), color='white')
pixels3 = np.zeros((512, 512, 3), dtype=np.uint8)

# Create checkerboard pattern with high frequency
for i in range(0, 512, 4):
    for j in range(0, 512, 4):
        if (i // 4 + j // 4) % 2 == 0:
            pixels3[i:i+4, j:j+4] = [0, 0, 0]
        else:
            pixels3[i:i+4, j:j+4] = [255, 255, 255]

# Add some sine wave patterns
x = np.linspace(0, 4 * np.pi, 512)
y = np.linspace(0, 4 * np.pi, 512)
X, Y = np.meshgrid(x, y)
Z = np.sin(X * 10) * np.cos(Y * 10) * 127 + 128

pixels3[:, :, 0] = np.clip(pixels3[:, :, 0] + Z / 4, 0, 255)
img3 = Image.fromarray(pixels3.astype(np.uint8))
img3.save("test_images/artificial_pattern.png")
print("✓ Saved: artificial_pattern.png")

# 4. Create a simple face image (for face detection test)
print("Creating simple face illustration...")
img4 = Image.new('RGB', (512, 512), color='#FDBCB4')  # Skin tone
draw4 = ImageDraw.Draw(img4)

# Draw simple face features
# Eyes
draw4.ellipse([150, 180, 200, 230], fill='white', outline='black', width=2)
draw4.ellipse([312, 180, 362, 230], fill='white', outline='black', width=2)
# Pupils
draw4.ellipse([165, 195, 185, 215], fill='black')
draw4.ellipse([327, 195, 347, 215], fill='black')
# Nose
draw4.polygon([(256, 250), (236, 290), (276, 290)], outline='black', width=2)
# Mouth
draw4.arc([200, 300, 312, 350], start=0, end=180, fill='red', width=3)
# Face outline
draw4.ellipse([100, 100, 412, 420], outline='black', width=3)

img4.save("test_images/simple_face.png")
print("✓ Saved: simple_face.png")

# 5. Create a composite/edited image (high suspicion expected)
print("Creating edited composite image...")
img5 = Image.new('RGB', (512, 512), color='lightblue')
draw5 = ImageDraw.Draw(img5)

# Background pattern
for i in range(0, 512, 20):
    draw5.line([(0, i), (512, i)], fill='gray', width=1)
    draw5.line([(i, 0), (i, 512)], fill='gray', width=1)

# Add multiple elements with different compression
box1 = Image.new('RGB', (150, 150), color='red')
img5.paste(box1, (50, 50))

# Save and reload with different quality
img5.save("test_images/temp.jpg", quality=90)
img5 = Image.open("test_images/temp.jpg")

# Add more elements
box2 = Image.new('RGB', (100, 100), color='green')
img5.paste(box2, (300, 300))

# Final save with different compression
img5.save("test_images/edited_composite.jpg", quality=60)
os.remove("test_images/temp.jpg")
print("✓ Saved: edited_composite.jpg")

print("\n" + "="*50)
print("✅ Test images created successfully!")
print("="*50)
print("\nTest images created in 'test_images' directory:")
print("1. authentic_gradient.jpg - Expected: Low suspicion")
print("2. compressed_shapes.jpg - Expected: High ELA score")
print("3. artificial_pattern.png - Expected: High FFT score")
print("4. simple_face.png - Expected: Face detected")
print("5. edited_composite.jpg - Expected: High suspicion")
print("\n📁 Location: " + os.path.abspath("test_images"))
