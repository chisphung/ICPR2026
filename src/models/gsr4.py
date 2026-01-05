"""
GSR4 (Geometric k-nearest neighbors Super-Resolution) Module.

Implements the warping and aggregation step to produce the restored image
by aligning neighboring frames and averaging with outlier rejection.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

from .optical_flow import warp_flow


class GSR4Aggregator(nn.Module):
    """
    GSR4 frame aggregation for super-resolution.
    
    Warps neighboring frames to the center frame coordinate system
    and aggregates using k-nearest neighbor averaging with outlier rejection.
    """
    
    def __init__(self, k_neighbors: int = 1, upscale_factor: int = 2):
        """
        Initialize GSR4 aggregator.
        
        Args:
            k_neighbors: Number of nearest neighbors per frame (default: 1)
            upscale_factor: Upscaling factor for super-resolution (default: 2)
        """
        super().__init__()
        self.k_neighbors = k_neighbors
        self.upscale_factor = upscale_factor
    
    def warp_frames_to_center(
        self,
        neighbor_frames: List[torch.Tensor],
        flows_to_center: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """
        Warp all neighboring frames to center frame coordinates.
        
        Args:
            neighbor_frames: List of frames (B, C, H, W)
            flows_to_center: List of optical flows from each neighbor to center
            
        Returns:
            List of warped frames aligned to center
        """
        warped_frames = []
        
        for frame, flow in zip(neighbor_frames, flows_to_center):
            warped = warp_flow(frame, flow)
            warped_frames.append(warped)
        
        return warped_frames
    
    def aggregate_frames(
        self,
        center_frame: torch.Tensor,
        warped_frames: List[torch.Tensor],
        method: str = "robust_mean",
    ) -> torch.Tensor:
        """
        Aggregate center and warped frames into restored image.
        
        Args:
            center_frame: Center reference frame (B, C, H, W)
            warped_frames: List of warped neighboring frames
            method: Aggregation method ('mean', 'median', 'robust_mean')
            
        Returns:
            Aggregated restored frame (B, C, H, W)
        """
        # Stack all frames including center
        all_frames = [center_frame] + warped_frames
        stacked = torch.stack(all_frames, dim=0)  # (N, B, C, H, W)
        
        if method == "mean":
            # Simple average
            result = stacked.mean(dim=0)
            
        elif method == "median":
            # Median for robustness to outliers
            result = stacked.median(dim=0)[0]
            
        elif method == "robust_mean":
            # Robust mean: exclude outliers before averaging
            result = self._robust_mean(stacked)
        
        else:
            result = stacked.mean(dim=0)
        
        return result
    
    def _robust_mean(
        self,
        stacked: torch.Tensor,
        trim_ratio: float = 0.2,
    ) -> torch.Tensor:
        """
        Compute robust mean by trimming outliers.
        
        Args:
            stacked: Stacked frames (N, B, C, H, W)
            trim_ratio: Fraction of extreme values to exclude
            
        Returns:
            Robust mean (B, C, H, W)
        """
        N, B, C, H, W = stacked.shape
        
        if N <= 2:
            return stacked.mean(dim=0)
        
        # Number of values to trim from each end
        k = max(1, int(N * trim_ratio / 2))
        
        # Sort along the frame dimension
        sorted_frames, _ = torch.sort(stacked, dim=0)
        
        # Keep middle values (exclude k from each end)
        if k >= N // 2:
            # Too few frames, just take median
            return stacked.median(dim=0)[0]
        
        trimmed = sorted_frames[k:N-k, ...]
        
        return trimmed.mean(dim=0)
    
    def upscale(
        self,
        image: torch.Tensor,
        mode: str = "bilinear",
    ) -> torch.Tensor:
        """
        Upscale image by the configured factor.
        
        Args:
            image: Input image (B, C, H, W)
            mode: Interpolation mode
            
        Returns:
            Upscaled image (B, C, H*factor, W*factor)
        """
        if self.upscale_factor == 1:
            return image
        
        return F.interpolate(
            image,
            scale_factor=self.upscale_factor,
            mode=mode,
            align_corners=True if mode != "nearest" else None,
        )
    
    def forward(
        self,
        center_frame: torch.Tensor,
        neighbor_frames: List[torch.Tensor],
        flows_to_center: List[torch.Tensor],
        upscale: bool = False,
    ) -> torch.Tensor:
        """
        Full GSR4 aggregation pipeline.
        
        Args:
            center_frame: Center frame (B, C, H, W)
            neighbor_frames: List of neighboring frames
            flows_to_center: Optical flows from neighbors to center
            upscale: Whether to upscale the result
            
        Returns:
            Restored frame (B, C, H', W')
        """
        # Warp neighbors to center
        warped = self.warp_frames_to_center(neighbor_frames, flows_to_center)
        
        # Aggregate
        restored = self.aggregate_frames(center_frame, warped, method="robust_mean")
        
        # Optionally upscale
        if upscale:
            restored = self.upscale(restored)
        
        return restored


class AdaptiveGSR4(GSR4Aggregator):
    """
    Adaptive GSR4 with learned aggregation weights.
    
    Uses a small CNN to predict per-pixel weights for frame aggregation.
    """
    
    def __init__(self, k_neighbors: int = 1, upscale_factor: int = 2, in_channels: int = 3):
        super().__init__(k_neighbors, upscale_factor)
        
        # Weight prediction network
        self.weight_net = nn.Sequential(
            nn.Conv2d(in_channels * 2, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid(),
        )
    
    def compute_weights(
        self,
        center_frame: torch.Tensor,
        warped_frame: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute aggregation weight for a warped frame.
        
        Args:
            center_frame: Reference frame (B, C, H, W)
            warped_frame: Warped neighbor frame (B, C, H, W)
            
        Returns:
            Weight map (B, 1, H, W)
        """
        concat = torch.cat([center_frame, warped_frame], dim=1)
        weight = self.weight_net(concat)
        return weight
    
    def aggregate_frames(
        self,
        center_frame: torch.Tensor,
        warped_frames: List[torch.Tensor],
        method: str = "weighted",
    ) -> torch.Tensor:
        """
        Weighted frame aggregation.
        """
        if method != "weighted" or not self.training:
            return super().aggregate_frames(center_frame, warped_frames, "robust_mean")
        
        # Compute weights for each warped frame
        weights = [torch.ones_like(center_frame[:, :1, :, :])]  # Center weight = 1
        
        for warped in warped_frames:
            w = self.compute_weights(center_frame, warped)
            weights.append(w)
        
        # Stack and normalize weights
        all_weights = torch.stack(weights, dim=0)  # (N, B, 1, H, W)
        all_weights = all_weights / (all_weights.sum(dim=0, keepdim=True) + 1e-8)
        
        # Stack frames
        all_frames = torch.stack([center_frame] + warped_frames, dim=0)  # (N, B, C, H, W)
        
        # Weighted sum
        result = (all_frames * all_weights).sum(dim=0)
        
        return result
