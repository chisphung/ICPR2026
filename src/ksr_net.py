"""
KSR-Net: Kinematic-Structural Refinement Network.

End-to-end pipeline for multi-frame license plate restoration and recognition.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

from .config import MFLPRConfig, DEFAULT_CONFIG
from .data.dataset import load_single_track, frames_to_tensor, tensor_to_numpy
from .models.hybrid_alignment import HybridAlignmentModule
from .models.video_swin import VideoSwinTransformer
from .models.optical_flow import OpticalFlowEstimator
from .models.recognizer import LicensePlateRecognizer


class ReconstructionHead(nn.Module):
    """
    Reconstruction head with PixelShuffle upsampling.
    """
    
    def __init__(
        self,
        in_channels: int = 96,
        out_channels: int = 3,
        upscale_factor: int = 2,
    ):
        super().__init__()
        
        # Feature refinement
        self.refine = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Upsampling with PixelShuffle
        self.upsample = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * upscale_factor ** 2, 3, 1, 1),
            nn.PixelShuffle(upscale_factor),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Output projection
        self.out = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels // 2, out_channels, 3, 1, 1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generate HR output from features."""
        x = self.refine(x)
        x = self.upsample(x)
        x = self.out(x)
        return torch.sigmoid(x)


class KSRNet(nn.Module):
    """
    Kinematic-Structural Refinement Network.
    
    Architecture:
    1. Hybrid Alignment Module (HAM): Flow + Deformable Attention
    2. Video Swin Transformer (STFT): Spatio-temporal fusion
    3. Reconstruction Head: Upsampling to HR
    4. Recognition Head: PARSeq/MGP-STR
    """
    
    def __init__(
        self,
        config: Optional[MFLPRConfig] = None,
        feat_channels: int = 64,
        embed_dim: int = 96,
    ):
        super().__init__()
        self.config = config or DEFAULT_CONFIG
        
        # 1. Optical Flow Estimator (for coarse alignment)
        self.flow_estimator = OpticalFlowEstimator(
            method="flowformer++",
            weights_path=self.config.flowformer_weights,
            device=self.config.device,
        )
        
        # 2. Hybrid Alignment Module
        self.alignment = HybridAlignmentModule(
            feat_channels=feat_channels,
            use_deformable=True,
        )
        
        # 3. Video Swin Transformer
        self.fusion = VideoSwinTransformer(
            in_channels=feat_channels,
            embed_dim=embed_dim,
            depths=(2, 2),
            num_heads=(3, 6),
            window_size=(2, 4, 4),
        )
        
        # 4. Reconstruction Head
        self.reconstruction = ReconstructionHead(
            in_channels=embed_dim,
            out_channels=3,
            upscale_factor=2,
        )
        
        # 5. Recognition Head
        self.recognizer = LicensePlateRecognizer(
            model_name=self.config.mgp_str_model,
            device=self.config.device,
        )
    
    def align_frames(
        self,
        frames: torch.Tensor,
        center_idx: int = 2,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Align all frames to center frame.
        
        Args:
            frames: Input frames (N, C, H, W)
            center_idx: Index of center frame
            
        Returns:
            Tuple of (aligned_features, aligned_frames)
        """
        N, C, H, W = frames.shape
        
        # Extract center and neighbors
        center = frames[center_idx].unsqueeze(0)
        neighbors = [frames[i].unsqueeze(0) for i in range(N) if i != center_idx]
        
        # Compute optical flows
        _, backward_flows = self.flow_estimator.compute_bidirectional_flow(
            center, neighbors
        )
        
        # Apply hybrid alignment
        aligned_feat, aligned_frames = self.alignment(
            center, neighbors, backward_flows
        )
        
        return aligned_feat, aligned_frames
    
    def forward(
        self,
        frames: torch.Tensor,
        center_idx: int = 2,
    ) -> Tuple[torch.Tensor, List[str]]:
        """
        Full KSR-Net forward pass.
        
        Args:
            frames: Input LR frames (N, C, H, W) or (B, N, C, H, W)
            center_idx: Index of center frame
            
        Returns:
            Tuple of (restored_HR, predicted_text)
        """
        # Handle batch dimension
        if frames.dim() == 4:
            frames = frames.unsqueeze(0)  # Add batch dim
        
        B, N, C, H, W = frames.shape
        
        all_restored = []
        all_predictions = []
        
        for b in range(B):
            # Get frames for this batch
            batch_frames = frames[b]  # (N, C, H, W)
            
            # 1. Align frames
            aligned_feat, _ = self.align_frames(batch_frames, center_idx)
            
            # 2. Prepare for Video Swin (need temporal dimension)
            # Stack center frame features with aligned features
            center_feat = self.alignment.feat_encoder(batch_frames[center_idx].unsqueeze(0))
            
            # Create temporal volume (B=1, C, T, H, W)
            feat_volume = aligned_feat.unsqueeze(2).repeat(1, 1, N, 1, 1)
            
            # 3. Spatio-temporal fusion
            fused = self.fusion(feat_volume)
            
            # 4. Reconstruction
            restored = self.reconstruction(fused)
            all_restored.append(restored)
            
            # 5. Recognition
            predictions = self.recognizer.recognize(restored)
            all_predictions.extend(predictions)
        
        # Stack results
        restored = torch.cat(all_restored, dim=0)
        
        return restored, all_predictions
    
    def restore(
        self,
        frames: torch.Tensor,
        center_idx: int = 2,
    ) -> torch.Tensor:
        """
        Restore HR image without recognition.
        
        Args:
            frames: Input LR frames (N, C, H, W)
            center_idx: Center frame index
            
        Returns:
            Restored HR image (1, C, H', W')
        """
        restored, _ = self.forward(frames, center_idx)
        return restored


class KSRNetPipeline:
    """
    Inference pipeline for KSR-Net.
    """
    
    def __init__(self, config: Optional[MFLPRConfig] = None):
        self.config = config or DEFAULT_CONFIG
        self.model = None
        self._loaded = False
    
    def load_model(self):
        """Load KSR-Net model."""
        if self._loaded:
            return
        
        self.model = KSRNet(self.config)
        self.model = self.model.to(self.config.device)
        self.model.eval()
        self._loaded = True
        print("KSR-Net loaded successfully!")
    
    @torch.no_grad()
    def process_track(self, track_path: Path) -> Dict:
        """
        Process a single track.
        
        Args:
            track_path: Path to track directory
            
        Returns:
            Dict with results
        """
        if not self._loaded:
            self.load_model()
        
        # Load data
        data = load_single_track(track_path)
        
        # Convert to tensor
        lr_frames = frames_to_tensor(data['lr_frames'], device=self.config.device)
        
        # Run KSR-Net
        restored, predictions = self.model(lr_frames)
        
        # Convert to numpy
        restored_np = tensor_to_numpy(restored[0])
        
        return {
            'track_path': str(track_path),
            'ground_truth': data['plate_text'],
            'prediction': predictions[0] if predictions else '',
            'restored_image': restored_np,
        }
