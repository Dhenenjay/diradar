# 🛰️ Deepfake Intelligence Radar - Verification Guide

## 🚀 Quick Start

### 1. Start the Application
```bash
# Option A: PowerShell (Windows)
.\bootstrap.ps1

# Option B: Bash (Linux/Mac/WSL)
./bootstrap.sh

# Option C: Direct Python
python -m streamlit run app.py --server.port 7860 --server.headless true
```

### 2. Access the Application
- **Local**: http://localhost:7860
- **Daytona**: Check your Daytona dashboard for the URL
- **Docker**: http://localhost:7860

## 📝 Test Procedure

### Step 1: Upload Test Images
1. Click "Browse files" in the file uploader
2. Navigate to `test_images` directory
3. Select all 5 test images (or test one at a time)
4. Click "Open"

### Step 2: Run Analysis
1. Verify files are listed in "View uploaded files"
2. Click the blue "🔍 Analyze" button
3. Wait for processing (should take 5-10 seconds)

### Step 3: Verify Results

#### Expected Results Table:

| Image | ELA Score | FFT Score | Face | Suspicion | Expected Behavior |
|-------|-----------|-----------|------|-----------|-------------------|
| **authentic_gradient.jpg** | Low (<0.05) | Low (<0.3) | ❌ No | 🟢 Low (<0.3) | Natural image pattern |
| **compressed_shapes.jpg** | High (>0.1) | Medium | ❌ No | 🟡-🟠 Medium (0.4-0.7) | Heavy compression artifacts |
| **artificial_pattern.png** | Medium | High (>0.6) | ❌ No | 🟠 High (0.6-0.8) | Artificial frequency patterns |
| **simple_face.png** | Low | Low | ✅ Yes | 🟡 Medium (0.3-0.5) | Face detected, boosts score |
| **edited_composite.jpg** | High | Medium | ❌ No | 🟠-🔴 High (0.6-0.9) | Multiple compression levels |

### Step 4: Check ELA Heatmaps
- Scroll down to "🔥 ELA Heatmap Analysis"
- Verify heatmaps show:
  - **authentic_gradient.jpg**: Mostly dark/uniform
  - **compressed_shapes.jpg**: Bright edges around shapes
  - **artificial_pattern.png**: Pattern visibility
  - **simple_face.png**: Face outline visible
  - **edited_composite.jpg**: Different brightness regions

### Step 5: Export Results
1. Click "📥 Download Results as CSV"
2. Open the CSV file
3. Verify all scores are present and formatted correctly

## 🔍 What to Look For

### ✅ Success Indicators:
- All images process without errors
- Scores align with expected ranges
- Face detection works on simple_face.png
- ELA heatmaps display correctly
- CSV export contains all data

### ⚠️ Common Issues:
- **Face not detected on simple drawing**: MediaPipe requires realistic faces
- **Scores vary slightly**: Normal due to compression/processing
- **Slow processing**: First run loads models, subsequent runs are faster

## 📊 Score Interpretation

### Suspicion Score Ranges:
- **0.0 - 0.3**: 🟢 Low suspicion (likely authentic)
- **0.3 - 0.6**: 🟡 Moderate suspicion (possible manipulation)
- **0.6 - 0.8**: 🟠 High suspicion (likely manipulated)
- **0.8 - 1.0**: 🔴 Very high suspicion (strong evidence)

### Individual Metrics:
- **ELA Score**: Compression inconsistencies (0-1)
- **FFT Score**: Frequency anomalies (0-1)
- **Face Detection**: Binary (0 or 1)

## 🎯 Advanced Testing

### Test with Real Images:
1. **Authentic photos**: Should score low (<0.3)
2. **AI-generated images**: Should score high (>0.6)
3. **Edited photos**: Should score medium-high (0.4-0.8)
4. **Deepfakes**: Should score very high (>0.7) with face detected

### Performance Testing:
- Upload 10+ images simultaneously
- Test different image sizes
- Test various formats (JPG, PNG)

## 📞 Troubleshooting

### App Won't Start:
```bash
# Check Python version
python --version  # Should be 3.10+

# Reinstall requirements
pip install -r requirements.txt --force-reinstall
```

### Analysis Fails:
- Check image formats (JPG/PNG only)
- Ensure files aren't corrupted
- Try smaller images (<5MB)

### Port Already in Use:
```bash
# Windows: Find and kill process
netstat -ano | findstr :7860
taskkill /PID <PID> /F

# Linux/Mac:
lsof -i :7860
kill -9 <PID>
```

## ✨ Success Criteria

Your Deepfake Intelligence Radar is working correctly if:
1. ✅ All 5 test images process successfully
2. ✅ Scores match expected ranges (±20%)
3. ✅ Face detection identifies simple_face.png
4. ✅ ELA heatmaps display visual differences
5. ✅ CSV export downloads with all data

---

**Ready for your hackathon! 🚀**
