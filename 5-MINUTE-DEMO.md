# 🎬 DiRadar - 5 Minute Demo Script
## Deepfake Intelligence Radar: AI-Powered Image Authenticity Detection

---

## ⏱️ DEMO TIMELINE (5 Minutes)
- **0:00-0:30** - Introduction & Problem Statement
- **0:30-1:30** - Repository Tour & Tech Stack
- **1:30-2:30** - Live Product Demo
- **2:30-3:30** - Technical Deep Dive (Daft & Daytona)
- **3:30-4:30** - Results Analysis & Use Cases
- **4:30-5:00** - Wrap-up & Q&A Setup

---

## 📋 PART 1: INTRODUCTION (30 seconds)

### Opening Statement:
"Welcome to DiRadar - Deepfake Intelligence Radar, a fast visual triage tool that helps assess image authenticity in seconds. In today's world of AI-generated content and sophisticated image manipulation, we need tools that can quickly flag suspicious images for further investigation."

### The Problem We Solve:
- 🚨 **Challenge**: Newsrooms, investigators, and content moderators need to quickly assess hundreds of images
- ⚡ **Solution**: DiRadar provides instant visual and numerical analysis using multiple detection techniques
- 🎯 **Goal**: Not to replace human judgment, but to augment it with data-driven insights

---

## 🗂️ PART 2: REPOSITORY TOUR (1 minute)

### Navigate to: https://github.com/Dhenenjay/diradar

**"Let me show you the project structure and highlight our key technologies:"**

### Core Components:
```
📁 diradar/
├── 📄 app.py                 → "Streamlit UI - our user interface"
├── 📄 pipeline.py            → "Core detection engine with Daft integration"
├── 📄 requirements.txt       → "Minimal dependencies for fast setup"
├── 🐳 Dockerfile            → "Container support for easy deployment"
├── 📄 daytona.yaml          → "Cloud development environment config"
└── 📁 test_images/          → "Sample images for testing"
```

### 🔧 Tech Stack Highlights:

**1. DAFT (Data Analytics Framework) - getdaft.io**
```python
# In pipeline.py - Show this code:
# "Daft enables us to process multiple images in parallel with lazy evaluation"
def build_df(paths):
    df = daft.from_pydict({"path": paths})
    df = df.with_column("image_bytes", col("path").apply(read_image_bytes, return_dtype=DataType.binary()))
    # Parallel processing of ELA, FFT, and face detection
    return df
```

**2. DAYTONA - Cloud Development Platform**
```yaml
# In daytona.yaml - Show this:
# "Daytona provides instant cloud dev environments - no local setup needed!"
name: diradar
image: daytonaio/workspace-python:latest
ports:
  - 7860:7860  # Streamlit port
```

---

## 🖥️ PART 3: LIVE PRODUCT DEMO (1 minute)

### Starting the Application:
```powershell
# "Let's start the app - it's as simple as running our bootstrap script"
./bootstrap.ps1

# The app starts on http://localhost:7860
```

### Demo Flow:

1. **Open Browser**: http://localhost:7860
   - "Notice the clean, intuitive interface designed for rapid triage"

2. **Upload Test Images**:
   - "I'll upload our test images that include authentic photos, edited images, and AI-generated content"
   - Drag and drop multiple files from `test_images/` folder

3. **Click Analyze**:
   - "Watch as DiRadar processes all images simultaneously using Daft's parallel processing"
   - "Processing includes ELA, FFT analysis, and face detection"

4. **View Results**:
   - 📊 **Suspicion Score Table**: "Higher scores indicate higher likelihood of manipulation"
   - 🗺️ **ELA Heatmaps**: "Visual representation shows edited regions in bright colors"
   - 📥 **CSV Export**: "Download results for reporting or further analysis"

---

## 🔬 PART 4: TECHNICAL DEEP DIVE (1 minute)

### Where We Use DAFT (Data Processing):

**Show code from `pipeline.py`:**

```python
# "DAFT gives us distributed computing capabilities for batch processing"

@udf(return_dtype=DataType.binary())
def ela_bytes(image_bytes):
    # Error Level Analysis - runs in parallel across all images
    return png_bytes, ela_score

@udf(return_dtype=DataType.float64())
def fft_score(image_bytes):
    # Frequency analysis - detects unnatural patterns
    return high_freq_ratio

# The magic happens here - Daft orchestrates parallel execution:
df = df.with_columns([
    col("image_bytes").apply(ela_bytes),  # Parallel ELA
    col("image_bytes").apply(fft_score),  # Parallel FFT
    col("image_bytes").apply(face_landmark_conf)  # Parallel face detection
])
```

**Benefits of Daft:**
- ⚡ **Lazy Evaluation**: Only computes what's needed
- 🚀 **Parallel Processing**: Analyzes multiple images simultaneously
- 📈 **Scalable**: Can handle hundreds of images efficiently
- 🔄 **Streaming**: Processes large datasets without memory issues

### Where We Use DAYTONA (Development Environment):

**Show `.devcontainer.json` and `daytona.yaml`:**

```bash
# "Daytona enables instant development environments in the cloud"

# One command to create a fully configured workspace:
daytona create .

# Benefits:
# ✅ No local Python installation needed
# ✅ Consistent environment across team members
# ✅ Pre-configured with all dependencies
# ✅ Accessible from anywhere via browser
```

**Demo Daytona Workflow (if time permits):**
1. "A new developer can contribute in minutes, not hours"
2. "No 'works on my machine' problems"
3. "Perfect for hackathons and rapid prototyping"

---

## 📊 PART 5: RESULTS ANALYSIS & USE CASES (1 minute)

### Understanding the Scores:

**Suspicion Score Breakdown:**
- **0-30**: Likely authentic
- **30-60**: Moderate suspicion, needs review
- **60-100**: High suspicion, likely manipulated

### Detection Methods Explained:

1. **ELA (Error Level Analysis)**:
   - "Recompresses JPEG and finds compression inconsistencies"
   - "Edited areas show different compression artifacts"

2. **FFT (Fast Fourier Transform)**:
   - "Analyzes frequency patterns in images"
   - "AI-generated images often have unnatural frequency distributions"

3. **Face Detection**:
   - "Uses MediaPipe to detect facial landmarks"
   - "Helps identify deepfakes targeting people"

### Real-World Use Cases:

📰 **Newsrooms**: "Verify user-submitted content before publication"
🔍 **Investigations**: "Quickly triage evidence for authenticity"
📱 **Social Media**: "Flag potentially manipulated content"
🎓 **Education**: "Teach media literacy with visual examples"

---

## 🎯 PART 6: WRAP-UP (30 seconds)

### Key Takeaways:

1. **Fast & Visual**: "Get results in seconds, not minutes"
2. **Multiple Techniques**: "Combines ELA, FFT, and face detection for robust analysis"
3. **Powered by Modern Tech**:
   - **Daft**: "For scalable, parallel image processing"
   - **Daytona**: "For instant, consistent dev environments"
4. **Open Source**: "Free to use, modify, and contribute"

### Call to Action:

"DiRadar is not about replacing human judgment - it's about augmenting it with data. In an era of synthetic media, tools like this help maintain trust in visual content."

### Links & Resources:
- 🌐 **Repository**: https://github.com/Dhenenjay/diradar
- 📦 **Daft Documentation**: https://www.getdaft.io/
- ☁️ **Daytona Platform**: https://www.daytona.io/
- 🚀 **Try it now**: Clone and run with `./bootstrap.ps1` or `./bootstrap.sh`

---

## 💡 DEMO TIPS:

### Preparation Checklist:
- [ ] Have the app already running in background
- [ ] Pre-load test images in a folder
- [ ] Have GitHub repo open in another tab
- [ ] Keep this script visible on second monitor/phone

### Speaking Points to Emphasize:
1. **Speed**: "Processes multiple images in parallel"
2. **Transparency**: "Shows why it flagged an image, not just a score"
3. **Accessibility**: "No AI/ML expertise needed to use"
4. **Scalability**: "From single images to batch processing"

### Common Questions & Answers:

**Q: How accurate is it?**
A: "It's a triage tool - it flags images for human review, not definitive judgment. Accuracy depends on the type of manipulation."

**Q: Can it detect all deepfakes?**
A: "No single tool can detect all manipulations. DiRadar uses multiple techniques to catch different types of edits."

**Q: Why Daft instead of Pandas?**
A: "Daft provides lazy evaluation and true parallel processing, making it perfect for compute-intensive image analysis."

**Q: How does Daytona help?**
A: "Daytona eliminates setup friction - anyone can contribute to the project in minutes with a fully configured cloud environment."

---

## 🎬 DEMO SCRIPT (What to Say):

### 0:00 - Start
"Hi everyone! Today I'm excited to show you DiRadar - a tool that helps detect potentially manipulated images in seconds."

### 0:30 - Show Repository
"Let's start with the GitHub repository. As you can see, we've built this with modern technologies..."

### 1:30 - Launch App
"Now let me show you the actual product in action. I'll upload some test images..."

### 2:30 - Technical Explanation
"The magic happens through Daft's parallel processing. Each image goes through three analysis pipelines simultaneously..."

### 3:30 - Results Discussion
"Notice how the edited image shows bright spots in the ELA heatmap - these are the manipulated areas..."

### 4:30 - Closing
"DiRadar demonstrates how we can use modern data processing and cloud development tools to tackle real-world problems. Questions?"

---

## 🚀 POST-DEMO FOLLOW-UP:

Share these links with attendees:
1. Repository: https://github.com/Dhenenjay/diradar
2. One-click setup: "Just run bootstrap.ps1 or bootstrap.sh"
3. Cloud development: "Try it with Daytona for zero local setup"

**Thank you for watching! Let's build a more trustworthy digital world together.**
