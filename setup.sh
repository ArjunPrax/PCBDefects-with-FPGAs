#!/bin/bash
# Setup script for URECA FPGA AI project
# Run this script to set up the development environment

set -e  # Exit on error

echo "=========================================="
echo "URECA FPGA AI Project Setup"
echo "=========================================="

# Check Python version
echo "Checking Python version..."
python3 --version

# Create virtual environment
echo ""
echo "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "Virtual environment created"
else
    echo "Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install development requirements (not board-specific)
echo ""
echo "Installing Python dependencies for development..."
pip install -r requirements-dev.txt

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "To activate the virtual environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To deactivate, run:"
echo "  deactivate"
echo ""
echo "Next steps:"
echo "1. Run integration tests: python scripts/test_integration.py"
echo "2. Train the model: python models/train.py --epochs 5 --batch-size 16"
echo "3. Quantize: python quantization/quantize_model.py --checkpoint models/checkpoints/best_model.pth"
echo ""
echo "For ZCU104 deployment, see docs/QUICKSTART.md"
echo ""
