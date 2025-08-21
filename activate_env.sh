#!/bin/bash
# Activation script for satellite detection project virtual environment

echo "🛰️  Activating Satellite Detection Project Environment"
echo "=================================================="

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found. Please run 'python3 -m venv .venv' first."
    exit 1
fi

# Activate the virtual environment
source .venv/bin/activate

echo "✅ Virtual environment activated!"
echo "📦 Python version: $(python --version)"
echo "📍 Project directory: $(pwd)"
echo "�� Ready for satellite detection development!"
echo ""
echo "Available commands:"
echo "  python src/data/dataset_loader.py  # Test dataset loading"
echo "  pip install -r requirements.txt   # Install all dependencies"
echo "  deactivate                        # Exit virtual environment"
echo ""

# Set useful environment variables
export DATASET_ROOT="/home/tanman/datasets/playground/NAPA2_Audacity_v2_training"
export PROJECT_ROOT="$(pwd)"

echo "Environment variables set:"
echo "  DATASET_ROOT: $DATASET_ROOT"
echo "  PROJECT_ROOT: $PROJECT_ROOT"
