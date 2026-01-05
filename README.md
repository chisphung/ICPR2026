# KSR-Net: License Plate Recognition from Low-Resolution Video

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A deep learning framework for **multi-frame license plate restoration and recognition** from low-resolution surveillance video. Implements the **Kinematic-Structural Refinement Network (KSR-Net)** architecture for the ICPR 2026 Low-Resolution License Plate Recognition competition.

## 🎯 Overview

This project addresses the challenge of recognizing license plates from degraded video sequences featuring:

- Heavy motion blur
- Low spatial resolution (~17×33 pixels)
- Compression artifacts
- Variable lighting conditions

**Key Features:**

- Multi-frame super-resolution using FlowFormer++ optical flow
- Hybrid Alignment with deformable attention
- Video Swin Transformer for spatio-temporal fusion
- MGP-STR text recognition
- Task-driven losses for text legibility

## 📁 Project Structure

```
ICPR/
├── src/
│   ├── models/
│   │   ├── hybrid_alignment.py   # Flow-Guided Deformable Attention
│   │   ├── video_swin.py         # 3D Video Swin Transformer
│   │   ├── optical_flow.py       # FlowFormer++ wrapper
│   │   ├── losses.py             # Task-driven loss functions
│   │   ├── recognizer.py         # MGP-STR text recognition
│   │   └── ...
│   ├── data/
│   │   └── dataset.py            # Data loading utilities
│   ├── ksr_net.py                # Main KSR-Net architecture
│   ├── pipeline.py               # MF-LPR² baseline pipeline
│   └── config.py                 # Configuration
├── flowformer/                   # FlowFormer++ repository
├── flowformerpp_weights/         # Pretrained optical flow weights
├── dataset/                      # Training/test data
├── checkpoints/                  # Saved model weights
├── train_ksr.py                  # Training script
├── infer_ksr.py                  # KSR-Net inference
├── infer.py                      # MF-LPR² baseline inference
└── requirements.txt
```

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/ICPR.git
cd ICPR

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### FlowFormer++ Setup

Download pretrained weights:

```bash
mkdir -p flowformerpp_weights
# Place things_288960.pth in flowformerpp_weights/
```

### Training

```bash
# Quick test (1 epoch, 50 samples)
python3 train_ksr.py --epochs 1 --batch_size 2 --num_samples 50 --device cuda

# Full training
python3 train_ksr.py --epochs 50 --batch_size 4 --num_samples 5000 --device cuda
```

### Inference

```bash
# Single track
python3 infer_ksr.py --track dataset/train/Scenario-A/Brazilian/track_00001 --device cuda

# Batch evaluation
python3 infer_ksr.py --data_dir dataset/train --num_samples 100 --device cuda
```

## 🏗️ Architecture

### KSR-Net Pipeline

```
┌─────────────────┐
│ 5 LR Frames     │
└────────┬────────┘
         ▼
┌─────────────────┐
│ FlowFormer++    │  (Optical Flow Estimation)
└────────┬────────┘
         ▼
┌─────────────────┐
│ Hybrid Alignment│  (Flow + Deformable Attention)
└────────┬────────┘
         ▼
┌─────────────────┐
│ Video Swin      │  (3D Spatio-Temporal Fusion)
└────────┬────────┘
         ▼
┌─────────────────┐
│ Reconstruction  │  (PixelShuffle Upsampling)
└────────┬────────┘
         ▼
┌─────────────────┐
│ MGP-STR         │  (Text Recognition)
└────────┬────────┘
         ▼
   Predicted Text
```

### Loss Functions

| Loss             | Weight | Purpose                     |
| ---------------- | ------ | --------------------------- |
| Charbonnier      | 1.0    | Robust pixel reconstruction |
| Gradient Profile | 0.5    | Edge sharpness              |
| Perceptual       | 0.1    | Feature-level similarity    |
| Semantic STR     | 0.5    | Recognition-driven          |

## 📊 Dataset

The ICPR LRLPR dataset structure:

```
dataset/
├── train/
│   ├── Scenario-A/
│   │   ├── Brazilian/
│   │   │   ├── track_00001/
│   │   │   │   ├── lr-001.png ... lr-005.png
│   │   │   │   ├── hr-001.png ... hr-005.png
│   │   │   │   └── annotations.json
│   │   │   └── ...
│   │   └── Mercosur/
│   └── Scenario-B/
└── test/
```

## 📈 Results

| Method              | SSIM  | PSNR     | Char Acc |
| ------------------- | ----- | -------- | -------- |
| MF-LPR² (baseline)  | 0.166 | 11.45 dB | 3.0%     |
| KSR-Net (1 epoch)   | 0.036 | 10.96 dB | -        |
| KSR-Net (50 epochs) | TBD   | TBD      | TBD      |

_Note: KSR-Net requires ~50-100 epochs for convergence_

## 🔧 Configuration

Key hyperparameters in `src/config.py`:

```python
theta_temp = 10.0      # Temporal filter threshold
theta_spatial = 20.0   # Spatial refinement threshold
gsr4_k_neighbors = 4   # Number of frames for aggregation
```

## 📚 References

- [FlowFormer++](https://github.com/XiaoyuShi97/FlowFormerPlusPlus) - Optical flow estimation
- [MGP-STR](https://huggingface.co/alibaba-damo/mgp-str-base) - Text recognition
- [Video Swin Transformer](https://github.com/SwinTransformer/Video-Swin-Transformer) - Spatio-temporal fusion

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- ICPR 2026 Competition organizers
- FlowFormer++ authors
- MGP-STR/Alibaba DAMO Academy
