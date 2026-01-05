#!/usr/bin/env python3
"""
KSR-Net Inference Script.

Usage:
    python3 infer_ksr.py --track dataset/train/Scenario-A/Brazilian/track_00001
    python3 infer_ksr.py --data_dir dataset/train --num_samples 10
"""
import sys
import os
import argparse
from pathlib import Path
from typing import List, Optional
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import torch
from PIL import Image
from tqdm import tqdm

from src.config import MFLPRConfig, DEFAULT_CONFIG
from src.data.dataset import LicensePlateDataset, load_single_track, frames_to_tensor, tensor_to_numpy
from src.ksr_net import KSRNet, KSRNetPipeline
from src.evaluate import compute_ssim, compute_psnr, compute_character_accuracy


def parse_args():
    parser = argparse.ArgumentParser(description='KSR-Net Inference')
    
    # Data arguments
    parser.add_argument('--track', type=str, default=None, help='Single track path')
    parser.add_argument('--data_dir', type=str, default='dataset/train', help='Dataset directory')
    parser.add_argument('--scenarios', nargs='+', default=None, help='Scenarios to process')
    parser.add_argument('--plate_types', nargs='+', default=None, help='Plate types to process')
    parser.add_argument('--num_samples', type=int, default=None, help='Number of samples')
    
    # Model arguments
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--output_dir', type=str, default='outputs/ksr_net', help='Output directory')
    
    return parser.parse_args()


def run_inference():
    args = parse_args()
    
    print("Initializing KSR-Net...")
    
    # Create config
    config = MFLPRConfig(
        device=args.device,
        output_dir=Path(args.output_dir),
    )
    
    # Initialize model
    model = KSRNet(config)
    model = model.to(args.device)
    model.eval()
    print("KSR-Net loaded!")
    
    # Process single track or dataset
    if args.track:
        process_single_track(args.track, model, config)
    else:
        process_dataset(args, model, config)


def process_single_track(track_path: str, model: KSRNet, config: MFLPRConfig):
    """Process a single track."""
    track_path = Path(track_path)
    print(f"Processing: {track_path}")
    
    # Load data
    data = load_single_track(track_path)
    
    # Convert to tensor
    lr_frames = frames_to_tensor(data['lr_frames'], device=config.device)
    
    # Run KSR-Net
    with torch.no_grad():
        restored, predictions = model(lr_frames)
    
    # Convert to numpy
    restored_np = tensor_to_numpy(restored[0])
    
    # Get ground truth
    hr_frames = frames_to_tensor(data['hr_frames'], device=config.device)
    hr_center = tensor_to_numpy(hr_frames[2])  # Center HR frame
    
    # Resize restored to match HR for comparison
    restored_pil = Image.fromarray(restored_np)
    hr_h, hr_w = hr_center.shape[:2]
    restored_resized = np.array(restored_pil.resize((hr_w, hr_h), Image.Resampling.BILINEAR))
    
    # Compute metrics
    ssim = compute_ssim(restored_resized, hr_center)
    psnr = compute_psnr(restored_resized, hr_center)
    
    # Print results
    print(f"Ground Truth: {data['plate_text']}")
    print(f"Prediction:   {predictions[0] if predictions else ''}")
    print(f"SSIM: {ssim:.4f}")
    print(f"PSNR: {psnr:.2f} dB")
    
    # Save output
    output_dir = config.output_dir / track_path.name
    output_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(restored_np).save(output_dir / 'restored.png')


def process_dataset(args, model: KSRNet, config: MFLPRConfig):
    """Process multiple tracks from dataset."""
    print(f"Loading dataset from {args.data_dir}...")
    
    dataset = LicensePlateDataset(
        data_dir=args.data_dir,
        scenarios=args.scenarios,
        plate_types=args.plate_types,
    )
    
    print(f"Found {len(dataset)} tracks")
    
    # Limit samples if specified
    num_samples = args.num_samples if args.num_samples else len(dataset)
    num_samples = min(num_samples, len(dataset))
    
    print(f"Processing {num_samples} tracks...")
    
    # Collect metrics
    ssim_scores = []
    psnr_scores = []
    char_accuracies = []
    
    for i in tqdm(range(num_samples)):
        sample = dataset[i]
        
        try:
            # Get frames - sample contains lists, convert to tensors
            lr_frames = frames_to_tensor(sample['lr_frames'], device=config.device)
            hr_frames = frames_to_tensor(sample['hr_frames'], device=config.device)
            
            # Run KSR-Net
            with torch.no_grad():
                restored, predictions = model(lr_frames)
            
            # Convert to numpy
            restored_np = tensor_to_numpy(restored[0])
            hr_center = tensor_to_numpy(hr_frames[2])
            
            # Resize for comparison
            restored_pil = Image.fromarray(restored_np)
            hr_h, hr_w = hr_center.shape[:2]
            restored_resized = np.array(restored_pil.resize((hr_w, hr_h), Image.Resampling.BILINEAR))
            
            # Metrics
            ssim = compute_ssim(restored_resized, hr_center)
            psnr = compute_psnr(restored_resized, hr_center)
            
            ssim_scores.append(ssim)
            psnr_scores.append(psnr)
            
            # Character accuracy
            gt_text = sample['plate_text']
            pred_text = predictions[0] if predictions else ''
            char_acc = compute_character_accuracy(pred_text, gt_text)
            char_accuracies.append(char_acc)
            
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue
    
    # Print report
    print("\n" + "=" * 50)
    print("KSR-Net Evaluation Report")
    print("=" * 50)
    print(f"Number of samples: {len(ssim_scores)}")
    print("-" * 50)
    print("Image Quality Metrics:")
    print(f"  SSIM:  {np.mean(ssim_scores):.4f} ± {np.std(ssim_scores):.4f}")
    print(f"  PSNR:  {np.mean(psnr_scores):.2f} ± {np.std(psnr_scores):.2f} dB")
    print("-" * 50)
    print("Recognition Metrics:")
    print(f"  Character Accuracy: {np.mean(char_accuracies) * 100:.2f}%")
    print("=" * 50)


if __name__ == '__main__':
    run_inference()
