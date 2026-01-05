#!/bin/bash
# Setup script for KSR-Net License Plate Recognition

set -e

echo "============================================"
echo "KSR-Net Setup Script"
echo "============================================"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Clone FlowFormer++ if not exists
if [ ! -d "flowformer" ]; then
    echo "Cloning FlowFormer++..."
    git clone https://github.com/XiaoyuShi97/FlowFormerPlusPlus.git flowformer
fi

# Create directories
echo "Creating directories..."
mkdir -p flowformerpp_weights
mkdir -p checkpoints
mkdir -p outputs
mkdir -p dataset

echo ""
echo "============================================"
echo "Setup Complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo "1. Download FlowFormer++ weights:"
echo "   - Place 'things_288960.pth' in flowformerpp_weights/"
echo ""
echo "2. Prepare dataset:"
echo "   - Place training data in dataset/train/"
echo ""
echo "3. Run training:"
echo "   python3 train_ksr.py --epochs 50 --batch_size 4 --device cuda"
echo ""
echo "4. Run inference:"
echo "   python3 infer_ksr.py --track dataset/train/Scenario-A/Brazilian/track_00001"
echo ""
