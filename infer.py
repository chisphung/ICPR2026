#!/usr/bin/env python3
"""
MF-LPR² Inference Script.

Run the complete restoration and recognition pipeline on license plate tracks.
"""
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from src.config import MFLPRConfig
from src.data.dataset import LicensePlateDataset, load_single_track, frames_to_tensor, tensor_to_numpy
from src.pipeline import MFLPRPipeline
from src.evaluate import evaluate_track, evaluate_dataset, print_evaluation_report


def run_single_track(
    pipeline: MFLPRPipeline,
    track_path: Path,
    output_dir: Path,
    device: str = "cuda",
) -> dict:
    """Run inference on a single track."""
    # Load data
    data = load_single_track(track_path)
    
    # Convert to tensor
    lr_frames = frames_to_tensor(data['lr_frames'], device=device)
    
    # Run pipeline
    with torch.no_grad():
        restored, predictions, info = pipeline(lr_frames, data['corners'])
    
    # Convert to numpy
    restored_np = tensor_to_numpy(restored[0])
    
    # Get ground truth HR image (center frame)
    center_idx = 2  # Middle frame
    gt_image = data['hr_frames'][center_idx] if data['hr_frames'] else None
    
    # Evaluate
    result = {
        'track_path': str(track_path),
        'ground_truth_text': data['plate_text'],
        'predicted_text': predictions[0] if predictions else '',
        'num_valid_frames': info['num_valid_frames'],
    }
    
    if gt_image is not None:
        eval_result = evaluate_track(
            restored_np, gt_image,
            predictions[0] if predictions else '',
            data['plate_text']
        )
        result.update(eval_result)
    
    # Save output
    if output_dir:
        track_out = output_dir / track_path.name
        track_out.mkdir(parents=True, exist_ok=True)
        
        Image.fromarray(restored_np).save(track_out / 'restored.png')
        
        with open(track_out / 'result.txt', 'w') as f:
            f.write(f"Ground Truth: {data['plate_text']}\n")
            f.write(f"Prediction: {predictions[0] if predictions else 'N/A'}\n")
            if 'ssim' in result:
                f.write(f"SSIM: {result['ssim']:.4f}\n")
                f.write(f"PSNR: {result['psnr']:.2f} dB\n")
    
    return result


def main():
    parser = argparse.ArgumentParser(description="MF-LPR² Inference")
    parser.add_argument(
        "--track", type=str, default=None,
        help="Path to a single track directory"
    )
    parser.add_argument(
        "--data_dir", type=str, default="dataset/train",
        help="Root data directory"
    )
    parser.add_argument(
        "--scenarios", type=str, nargs="+", default=["Scenario-A"],
        help="Scenarios to evaluate"
    )
    parser.add_argument(
        "--plate_types", type=str, nargs="+", default=["Brazilian", "Mercosur"],
        help="Plate types to evaluate"
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs",
        help="Output directory for results"
    )
    parser.add_argument(
        "--num_samples", type=int, default=None,
        help="Limit number of samples to process"
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use"
    )
    parser.add_argument(
        "--theta_temp", type=float, default=10.0,
        help="Temporal filtering threshold"
    )
    parser.add_argument(
        "--theta_spatial", type=float, default=20.0,
        help="Spatial refinement threshold"
    )
    args = parser.parse_args()
    
    # Setup config
    config = MFLPRConfig(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        device=args.device,
        theta_temp=args.theta_temp,
        theta_spatial=args.theta_spatial,
    )
    
    # Create output directory
    config.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize pipeline
    print("Initializing MF-LPR² pipeline...")
    pipeline = MFLPRPipeline(config)
    
    # Single track mode
    if args.track:
        print(f"Processing single track: {args.track}")
        result = run_single_track(
            pipeline,
            Path(args.track),
            config.output_dir,
            args.device,
        )
        print(f"Ground Truth: {result['ground_truth_text']}")
        print(f"Prediction:   {result['predicted_text']}")
        if 'ssim' in result:
            print(f"SSIM: {result['ssim']:.4f}")
            print(f"PSNR: {result['psnr']:.2f} dB")
        return
    
    # Batch mode
    print("Loading dataset...")
    dataset = LicensePlateDataset(
        config.data_dir,
        scenarios=args.scenarios,
        plate_types=args.plate_types,
    )
    
    if args.num_samples:
        tracks = dataset.tracks[:args.num_samples]
    else:
        tracks = dataset.tracks
    
    print(f"Processing {len(tracks)} tracks...")
    
    results = []
    for track_path in tqdm(tracks):
        try:
            result = run_single_track(
                pipeline,
                track_path,
                config.output_dir,
                args.device,
            )
            results.append(result)
        except Exception as e:
            print(f"Error processing {track_path}: {e}")
            continue
    
    # Aggregate and print results
    metrics = evaluate_dataset(results)
    print_evaluation_report(metrics)
    
    # Save summary
    with open(config.output_dir / 'summary.txt', 'w') as f:
        f.write(f"MF-LPR² Evaluation Summary\n")
        f.write(f"{'=' * 50}\n")
        f.write(f"Samples: {metrics.get('num_samples', 0)}\n")
        f.write(f"SSIM: {metrics.get('ssim_mean', 0):.4f} ± {metrics.get('ssim_std', 0):.4f}\n")
        f.write(f"PSNR: {metrics.get('psnr_mean', 0):.2f} ± {metrics.get('psnr_std', 0):.2f} dB\n")
        f.write(f"Character Accuracy: {metrics.get('char_accuracy_overall', 0)*100:.2f}%\n")
        f.write(f"Plate Accuracy: {metrics.get('plate_accuracy', 0)*100:.2f}%\n")


if __name__ == "__main__":
    main()
