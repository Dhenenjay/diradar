# Quick Public Deployment with ngrok
Write-Host "🌐 Deploying to Public URL..." -ForegroundColor Cyan

# Install pyngrok if needed
python -m pip install pyngrok -q

# Create Python script to run app with ngrok
@'
import subprocess
import sys
from pyngrok import ngrok

# Start Streamlit
process = subprocess.Popen([
    sys.executable, "-m", "streamlit", "run", "app.py",
    "--server.port", "7860",
    "--server.headless", "true"
])

# Create public tunnel
public_url = ngrok.connect(7860)
print(f"\n🌐 PUBLIC URL: {public_url}")
print(f"Share this URL for the hackathon demo!\n")

# Keep running
try:
    process.wait()
except KeyboardInterrupt:
    ngrok.disconnect(public_url)
    process.terminate()
'@ | Out-File -FilePath run_public.py -Encoding UTF8

# Run the public deployment
python run_public.py
