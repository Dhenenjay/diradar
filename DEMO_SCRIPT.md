# 🎬 Demo Recording Script

## 📹 Recording Setup
- **Screen Resolution**: 1920x1080 recommended
- **Recording Tool**: OBS Studio, Camtasia, or Windows Game Bar (Win+G)
- **Browser**: Chrome/Edge in clean profile
- **Duration**: Aim for 60-90 seconds

## 🎯 Demo Workflow (3 Images)

### 1. **Start Recording** (0:00)

### 2. **Show App Landing** (0:00-0:10)
- Start app: `python -m streamlit run app.py --server.port 7860 --server.headless true`
- Open browser to http://localhost:7860
- **PAUSE** on landing page for 3 seconds
- **HIGHLIGHT**: Point out "Built with Daft" and "Powered by Daytona" badges at the top

### 3. **Upload Images** (0:10-0:20)
- Click "Browse files"
- Navigate to `test_images/` folder
- Select these 3 images (for best demo effect):
  1. `authentic_gradient.jpg` (Low risk)
  2. `compressed_shapes.jpg` (Medium risk)
  3. `artificial_pattern.png` (High risk)
- Click "Open"
- **SHOW**: Green checkmark "✅ 3 file(s) uploaded"
- Click "View uploaded files" expander briefly to show file list

### 4. **Analyze Images** (0:20-0:30)
- Click the blue "🔍 Analyze" button
- **SHOW**: Loading spinner "Analyzing 3 image(s)..."
- Wait for completion (5-10 seconds)

### 5. **Show Results Table** (0:30-0:45)
- **PAUSE** on summary metrics (Average Suspicion, High Risk Images, etc.)
- **SCROLL** slowly through the results table
- **HIGHLIGHT** the color-coded Suspicion Score column:
  - Green = Low risk
  - Yellow/Orange = Medium risk
  - Red = High risk
- **POINT OUT**: "Face Detected" column shows ❌ or ✅

### 6. **Show ELA Heatmaps** (0:45-0:60)
- **SCROLL** down to "🔥 ELA Heatmap Analysis"
- **PAUSE** to show the 3 heatmaps side by side
- **HIGHLIGHT** differences:
  - Dark/uniform areas = original
  - Bright/hot areas = potential manipulation
- **POINT OUT** individual scores below each heatmap

### 7. **Highlight Tech Stack** (0:60-0:70)
- **SCROLL** to footer
- **HIGHLIGHT**: "Built with Daft DataFrame Engine + Daytona Dev Platform"
- **SCROLL** back to top
- **POINT OUT** the badges again

### 8. **Export Results** (0:70-0:80)
- **SCROLL** to results table
- Click "📥 Download Results as CSV"
- Show file downloaded notification

### 9. **End Scene** (0:80-0:90)
- Return to top of page
- **PAUSE** on title and tech stack badges
- Stop recording

## 📝 Narration Script (Optional Voice-Over)

**[0:00-0:10]**
"Welcome to Deepfake Intelligence Radar, a fast image triage tool built with Daft DataFrame Engine and powered by Daytona's cloud development platform."

**[0:10-0:20]**
"Let's analyze three test images for potential manipulation. I'll upload an authentic image, a compressed image, and one with artificial patterns."

**[0:20-0:30]**
"The system uses Daft's scalable UDF pipeline to process multiple analysis techniques in parallel."

**[0:30-0:45]**
"Here are our results. Notice the color-coded suspicion scores - green for low risk, orange for medium, and red for high risk. Each image gets scored on compression artifacts, frequency patterns, and face detection."

**[0:45-0:60]**
"The ELA heatmaps visualize potential manipulation areas. Bright regions indicate compression inconsistencies that might suggest editing."

**[0:60-0:70]**
"This entire pipeline leverages Daft for distributed processing and Daytona for seamless cloud development."

**[0:70-0:80]**
"Results can be exported as CSV for further analysis or reporting."

**[0:80-0:90]**
"Deepfake Intelligence Radar - Fast, visual triage for image authenticity. Built with Daft and Daytona."

## 🎨 Post-Production Tips

1. **Add Title Card** (optional):
   - "Deepfake Intelligence Radar"
   - "Built with Daft + Daytona"
   - Your hackathon team name

2. **Add Annotations**:
   - Arrow pointing to Daft/Daytona badges
   - Highlight box around suspicion scores
   - Text overlay for key features

3. **Export Settings**:
   - Format: MP4
   - Resolution: 1920x1080
   - Framerate: 30fps
   - Bitrate: 5000-8000 kbps

## ✅ Checklist Before Recording

- [ ] App is running on port 7860
- [ ] Test images are ready in `test_images/` folder
- [ ] Browser is clean (no personal bookmarks/tabs)
- [ ] Screen recorder is configured
- [ ] Microphone tested (if doing voice-over)
- [ ] Close unnecessary applications
- [ ] Set browser zoom to 100%

## 🚀 Quick Commands

Start the app:
```bash
python -m streamlit run app.py --server.port 7860 --server.headless true
```

Open in browser:
```
http://localhost:7860
```

---

**Good luck with your demo! 🎬**
