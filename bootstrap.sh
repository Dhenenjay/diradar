#!/bin/bash

# Deepfake Intelligence Radar - Bootstrap Script
# This script sets up the environment and launches the Streamlit app

echo "=========================================="
echo "🛰️  Deepfake Intelligence Radar Setup"
echo "=========================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "❌ Error: Python is not installed or not in PATH"
    echo "Please install Python 3.10 or higher and try again."
    exit 1
fi

# Determine Python command (python3 or python)
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    PYTHON_CMD="python"
fi

# Check Python version
echo "🐍 Checking Python version..."
PYTHON_VERSION=$($PYTHON_CMD -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null)

if [ $? -ne 0 ]; then
    echo "❌ Error: Could not determine Python version"
    exit 1
fi

echo "✅ Found Python $PYTHON_VERSION"

# Check if we're in a virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo ""
    echo "⚠️  Warning: Not running in a virtual environment"
    echo "It's recommended to use a virtual environment."
    echo ""
    read -p "Do you want to continue anyway? (y/n): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Exiting. Please activate a virtual environment and run again."
        exit 1
    fi
fi

# Install/upgrade pip
echo ""
echo "📦 Upgrading pip..."
$PYTHON_CMD -m pip install --upgrade pip --quiet

# Install requirements
echo ""
echo "📦 Installing requirements..."
echo "This may take a few minutes on first run..."

if [ -f "requirements.txt" ]; then
    $PYTHON_CMD -m pip install -r requirements.txt
    
    if [ $? -ne 0 ]; then
        echo ""
        echo "❌ Error: Failed to install requirements"
        echo "Please check the error messages above and try again."
        exit 1
    fi
else
    echo "❌ Error: requirements.txt not found"
    echo "Please ensure you're running this script from the project directory."
    exit 1
fi

echo ""
echo "✅ All requirements installed successfully!"

# Check if app.py exists
if [ ! -f "app.py" ]; then
    echo "❌ Error: app.py not found"
    echo "Please ensure you're running this script from the project directory."
    exit 1
fi

# Launch Streamlit app
echo ""
echo "=========================================="
echo "🚀 Launching Deepfake Intelligence Radar"
echo "=========================================="
echo ""
echo "📡 Server Configuration:"
echo "   - Port: 7860"
echo "   - Mode: Headless (no browser auto-open)"
echo ""
echo "🌐 Access the application at:"
echo "   - Local:    http://localhost:7860"
echo "   - Network:  http://$(hostname -I | awk '{print $1}'):7860" 2>/dev/null || echo "   - Network:  http://[your-ip]:7860"
echo ""
echo "📝 Logs:"
echo "----------------------------------------"

# Run Streamlit with specified configuration
$PYTHON_CMD -m streamlit run app.py \
    --server.port 7860 \
    --server.headless true \
    --server.address 0.0.0.0 \
    --browser.gatherUsageStats false \
    --theme.base "dark" \
    --theme.primaryColor "#FF6B6B" \
    --theme.backgroundColor "#0E1117" \
    --theme.secondaryBackgroundColor "#262730" \
    --theme.textColor "#FAFAFA"

# This will only run if Streamlit exits
echo ""
echo "🛑 Application stopped."
