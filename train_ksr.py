#!/usr/bin/env python3
"""
KSR-Net Training Script.

Three-stage training protocol:
1. Alignment Pre-training (optional)
2. Reconstruction Warm-up
3. End-to-End Fine-tuning

Usage:
    python3 train_ksr.py --epochs 1 --batch_size 4 --device cuda
"""
import sys
import os
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from src.config import MFLPRConfig, DEFAULT_CONFIG
from src.data.dataset import LicensePlateDataset, frames_to_tensor, tensor_to_numpy
from src.ksr_net import KSRNet
from src.models.losses import KSRNetLoss, CharbonnierLoss, GradientProfileLoss
from src.evaluate import compute_ssim, compute_psnr


class TrainingDataset(LicensePlateDataset):
    """Dataset wrapper that returns tensors instead of lists."""
    
    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        
        # Convert to tensors
        lr_frames = self._to_tensor(sample['lr_frames'])
        hr_frames = self._to_tensor(sample['hr_frames']) if sample['hr_frames'] else None
        
        return {
            'lr_frames': lr_frames,
            'hr_frames': hr_frames,
            'plate_text': sample['plate_text'],
            'track_path': sample['track_path'],
        }
    
    def _to_tensor(self, frames):
        """Convert list of numpy frames to tensor."""
        if frames is None:
            return None
        
        # Find max dimensions
        max_h = max(f.shape[0] for f in frames)
        max_w = max(f.shape[1] for f in frames)
        
        tensors = []
        for frame in frames:
            # Resize if needed
            if frame.shape[0] != max_h or frame.shape[1] != max_w:
                from PIL import Image
                img = Image.fromarray(frame)
                img = img.resize((max_w, max_h), Image.Resampling.BILINEAR)
                frame = np.array(img)
            
            # Convert to float and normalize
            frame = frame.astype(np.float32) / 255.0
            # HWC to CHW
            frame = np.transpose(frame, (2, 0, 1))
            tensors.append(torch.from_numpy(frame))
        
        return torch.stack(tensors)


def collate_fn(batch):
    """Custom collate function for variable size images."""
    lr_frames = []
    hr_frames = []
    texts = []
    paths = []
    
    for sample in batch:
        lr_frames.append(sample['lr_frames'])
        if sample['hr_frames'] is not None:
            hr_frames.append(sample['hr_frames'])
        texts.append(sample['plate_text'])
        paths.append(sample['track_path'])
    
    # Stack if all same size, otherwise keep as list
    try:
        lr_batch = torch.stack(lr_frames)
        hr_batch = torch.stack(hr_frames) if hr_frames else None
    except RuntimeError:
        # Variable sizes - use first sample's size
        lr_batch = lr_frames[0].unsqueeze(0)
        hr_batch = hr_frames[0].unsqueeze(0) if hr_frames else None
    
    return {
        'lr_frames': lr_batch,
        'hr_frames': hr_batch,
        'plate_text': texts,
        'track_path': paths,
    }


class KSRNetTrainer:
    """Trainer for KSR-Net."""
    
    def __init__(
        self,
        config: MFLPRConfig,
        model: KSRNet,
        train_loader: DataLoader,
        val_loader: DataLoader = None,
        lr: float = 1e-4,
        use_amp: bool = True,
    ):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.use_amp = use_amp
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=1e-4,
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=len(train_loader) * 10,  # 10 epochs
            eta_min=1e-7,
        )
        
        # Loss functions
        self.loss_fn = KSRNetLoss(
            lambda_pix=1.0,
            lambda_grad=0.5,
            lambda_percep=0.0,  # Disabled - AMP dtype issues
            lambda_sem=0.0,  # Disable semantic loss for now
        )
        
        # Mixed precision
        self.scaler = GradScaler() if use_amp else None
        
        # Metrics tracking
        self.train_losses = []
        self.val_metrics = []
    
    def train_epoch(self, epoch: int) -> dict:
        """Train for one epoch."""
        self.model.train()
        
        epoch_losses = []
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Get data
            lr_frames = batch['lr_frames'].to(self.config.device)
            hr_frames = batch['hr_frames'].to(self.config.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with autocast():
                    # Run model - get center HR frame as target
                    restored, _ = self.model(lr_frames)
                    
                    # Resize restored to match HR
                    # hr_frames shape: (B, N, C, H, W)
                    B, N_hr, C_hr, H_hr, W_hr = hr_frames.shape
                    restored_resized = nn.functional.interpolate(
                        restored, size=(H_hr, W_hr), mode='bilinear', align_corners=True
                    )
                    
                    # Target is center HR frame
                    target = hr_frames[:, 2]  # (B, C, H, W)
                    
                    # Compute loss
                    loss, loss_dict = self.loss_fn(restored_resized, target)
                
                # Backward with scaling
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Run model
                restored, _ = self.model(lr_frames)
                
                # Resize and get target
                # hr_frames shape: (B, N, C, H, W)
                B, N_hr, C_hr, H_hr, W_hr = hr_frames.shape
                restored_resized = nn.functional.interpolate(
                    restored, size=(H_hr, W_hr), mode='bilinear', align_corners=True
                )
                target = hr_frames[:, 2]
                
                # Loss
                loss, loss_dict = self.loss_fn(restored_resized, target)
                
                # Backward
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
            
            self.scheduler.step()
            
            # Track loss
            epoch_losses.append(loss.item())
            
            # Update progress
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}",
            })
        
        return {
            'epoch': epoch,
            'train_loss': np.mean(epoch_losses),
            'train_loss_std': np.std(epoch_losses),
        }
    
    @torch.no_grad()
    def validate(self) -> dict:
        """Validate the model."""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        
        ssim_scores = []
        psnr_scores = []
        
        for batch in tqdm(self.val_loader, desc="Validation"):
            lr_frames = batch['lr_frames'].to(self.config.device)
            hr_frames = batch['hr_frames'].to(self.config.device)
            
            # Forward
            restored, _ = self.model(lr_frames)
            
            # Resize
            # hr_frames shape: (B, N, C, H, W)
            B, N_hr, C_hr, H_hr, W_hr = hr_frames.shape
            restored_resized = nn.functional.interpolate(
                restored, size=(H_hr, W_hr), mode='bilinear', align_corners=True
            )
            
            # Compute metrics for each sample
            for i in range(restored_resized.shape[0]):
                pred = tensor_to_numpy(restored_resized[i])
                target = tensor_to_numpy(hr_frames[i, 2])
                
                ssim_scores.append(compute_ssim(pred, target))
                psnr_scores.append(compute_psnr(pred, target))
        
        return {
            'val_ssim': np.mean(ssim_scores),
            'val_ssim_std': np.std(ssim_scores),
            'val_psnr': np.mean(psnr_scores),
            'val_psnr_std': np.std(psnr_scores),
        }
    
    def save_checkpoint(self, path: Path, epoch: int, metrics: dict):
        """Save model checkpoint."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
        }, path)
        print(f"Saved checkpoint to {path}")


def parse_args():
    parser = argparse.ArgumentParser(description='Train KSR-Net')
    
    # Data
    parser.add_argument('--data_dir', type=str, default='dataset/train')
    parser.add_argument('--scenarios', nargs='+', default=['Scenario-A'])
    parser.add_argument('--plate_types', nargs='+', default=['Brazilian', 'Mercosur'])
    parser.add_argument('--val_split', type=float, default=0.1)
    parser.add_argument('--num_samples', type=int, default=None, help='Limit training samples')
    
    # Training
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Model
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--amp', action='store_true', default=True, help='Use mixed precision')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='checkpoints')
    parser.add_argument('--exp_name', type=str, default=None)
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup experiment name
    if args.exp_name is None:
        args.exp_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    output_dir = Path(args.output_dir) / args.exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("KSR-Net Training")
    print("=" * 60)
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Epochs: {args.epochs}")
    print(f"Output: {output_dir}")
    print("=" * 60)
    
    # Create config
    config = MFLPRConfig(
        device=args.device,
        output_dir=output_dir,
    )
    
    # Load dataset
    print("\nLoading dataset...")
    full_dataset = TrainingDataset(
        data_dir=args.data_dir,
        scenarios=args.scenarios,
        plate_types=args.plate_types,
        load_hr=True,
    )
    
    # Limit samples if specified
    if args.num_samples and args.num_samples < len(full_dataset):
        indices = list(range(args.num_samples))
        full_dataset = torch.utils.data.Subset(full_dataset, indices)
        print(f"Limited to {args.num_samples} samples")
    
    # Split train/val
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    # Initialize model
    print("\nInitializing KSR-Net...")
    model = KSRNet(config)
    model = model.to(args.device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create trainer
    trainer = KSRNetTrainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=args.lr,
        use_amp=args.amp,
    )
    
    # Training loop
    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60)
    
    best_ssim = 0.0
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_metrics = trainer.train_epoch(epoch)
        print(f"\nEpoch {epoch} - Train Loss: {train_metrics['train_loss']:.4f}")
        
        # Validate
        if len(val_loader) > 0:
            val_metrics = trainer.validate()
            print(f"  Val SSIM: {val_metrics.get('val_ssim', 0):.4f} ± {val_metrics.get('val_ssim_std', 0):.4f}")
            print(f"  Val PSNR: {val_metrics.get('val_psnr', 0):.2f} ± {val_metrics.get('val_psnr_std', 0):.2f} dB")
            
            # Save best model
            if val_metrics.get('val_ssim', 0) > best_ssim:
                best_ssim = val_metrics['val_ssim']
                trainer.save_checkpoint(
                    output_dir / 'best_model.pth',
                    epoch,
                    {**train_metrics, **val_metrics},
                )
        
        # Save latest
        trainer.save_checkpoint(
            output_dir / 'latest_model.pth',
            epoch,
            train_metrics,
        )
    
    # Final report
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Best Val SSIM: {best_ssim:.4f}")
    print(f"Checkpoints saved to: {output_dir}")
    
    # Save training summary
    summary_path = output_dir / 'training_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("KSR-Net Training Summary\n")
        f.write("=" * 40 + "\n")
        f.write(f"Epochs: {args.epochs}\n")
        f.write(f"Batch Size: {args.batch_size}\n")
        f.write(f"Learning Rate: {args.lr}\n")
        f.write(f"Train Samples: {len(train_dataset)}\n")
        f.write(f"Val Samples: {len(val_dataset)}\n")
        f.write(f"Best Val SSIM: {best_ssim:.4f}\n")
    
    print(f"Summary saved to: {summary_path}")


if __name__ == '__main__':
    main()
