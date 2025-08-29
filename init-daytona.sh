#!/bin/bash

# Daytona Sandbox Initialization Script
# This script sets up and launches a Daytona workspace for the Deepfake Intelligence Radar

echo "======================================"
echo "🚀 Daytona Workspace Setup"
echo "======================================"
echo ""

# Check if Daytona CLI is installed
if ! command -v daytona &> /dev/null; then
    echo "❌ Daytona CLI is not installed"
    echo ""
    echo "Please install it first:"
    echo "  curl -L https://download.daytona.io/daytona/install.sh | bash"
    exit 1
fi

echo "✅ Daytona CLI found"
echo ""

# Check if API key is configured
if ! daytona config get api-key &> /dev/null; then
    echo "⚠️  No Daytona API key configured"
    echo ""
    echo "Please configure your API key:"
    echo "  daytona config set api-key YOUR_API_KEY"
    echo ""
    echo "Then run this script again."
    exit 1
fi

echo "✅ API key configured"
echo ""

# Initialize workspace
echo "📦 Creating Daytona workspace..."
daytona create workspace \
    --name diradar \
    --repo . \
    --ide vscode \
    --image mcr.microsoft.com/devcontainers/python:3.10

# Wait for workspace to be ready
echo ""
echo "⏳ Waiting for workspace to initialize..."
sleep 5

# Connect to workspace
echo ""
echo "🔗 Connecting to workspace..."
daytona connect diradar

# Open workspace in browser
echo ""
echo "🌐 Opening workspace in browser..."
daytona open diradar

echo ""
echo "======================================"
echo "✅ Workspace Ready!"
echo "======================================"
echo ""
echo "To start the application inside the workspace:"
echo "  1. Open the terminal in the Daytona IDE"
echo "  2. Run: ./bootstrap.sh"
echo ""
echo "The app will be available at the forwarded port 7860"
echo ""
