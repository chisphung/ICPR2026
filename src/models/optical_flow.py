"""
Optical Flow Estimation Module.

Supports FlowFormer++ for high-quality optical flow estimation,
with fallback to correlation-based method for small images.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path


class OpticalFlowEstimator(nn.Module):
    """
    Optical flow estimation with FlowFormer++ support.
    
    For small license plate images, uses upscaling before FlowFormer++
    to get better flow estimates.
    """
    
    def __init__(
        self,
        method: str = "flowformer++",
        weights_path: Optional[Path] = None,
        device: str = "cuda",
        min_size: int = 128,  # Minimum image size for FlowFormer++
    ):
        """
        Initialize the optical flow estimator.
        
        Args:
            method: 'flowformer++' or 'correlation'
            weights_path: Path to FlowFormer++ weights
            device: Target device
            min_size: Minimum size to upscale images to for FlowFormer++
        """
        super().__init__()
        self.method = method
        self.device = device
        self.weights_path = weights_path
        self.min_size = min_size
        self.flowformer = None
        self._loaded = False
        
        if method == "flowformer++":
            self._init_flowformer()
    
    def _init_flowformer(self):
        """Initialize FlowFormer++ if available."""
        try:
            from .flowformer_wrapper import FlowFormerPPWrapper
            
            if self.weights_path and Path(self.weights_path).exists():
                # Get flowformer directory relative to weights
                flowformer_dir = Path(self.weights_path).parent.parent / "flowformer"
                
                self.flowformer = FlowFormerPPWrapper(
                    weights_path=self.weights_path,
                    device=self.device,
                    flowformer_dir=flowformer_dir,
                )
                print(f"FlowFormer++ initialized with weights: {self.weights_path}")
            else:
                print(f"FlowFormer++ weights not found at {self.weights_path}")
                print("Using correlation-based flow estimation")
                self.method = "correlation"
        except Exception as e:
            print(f"Failed to initialize FlowFormer++: {e}")
            import traceback
            traceback.print_exc()
            self.method = "correlation"
    
    def upscale_image(self, img: torch.Tensor, target_size: int) -> Tuple[torch.Tensor, float]:
        """
        Upscale image to minimum size for FlowFormer++.
        
        Args:
            img: Input image (B, C, H, W)
            target_size: Target minimum dimension
            
        Returns:
            Tuple of (upscaled_image, scale_factor)
        """
        H, W = img.shape[-2:]
        
        if min(H, W) >= target_size:
            return img, 1.0
        
        # Calculate scale factor to reach target size
        scale = target_size / min(H, W)
        new_H = int(H * scale)
        new_W = int(W * scale)
        
        # Upscale
        upscaled = F.interpolate(
            img, size=(new_H, new_W), mode='bilinear', align_corners=True
        )
        
        return upscaled, scale
    
    def downscale_flow(self, flow: torch.Tensor, scale: float, orig_size: Tuple[int, int]) -> torch.Tensor:
        """
        Downscale flow back to original size and adjust magnitude.
        
        Args:
            flow: Optical flow (B, 2, H', W')
            scale: Scale factor used for upscaling
            orig_size: Original (H, W)
            
        Returns:
            Downscaled flow (B, 2, H, W)
        """
        if scale == 1.0:
            return flow
        
        H, W = orig_size
        
        # Downscale flow
        flow_down = F.interpolate(
            flow, size=(H, W), mode='bilinear', align_corners=True
        )
        
        # Adjust flow magnitude (flow values are in pixels)
        flow_down = flow_down / scale
        
        return flow_down
    
    def compute_flow(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute optical flow from source to target.
        
        Args:
            source: Source frame (B, C, H, W)
            target: Target frame (B, C, H, W)
            
        Returns:
            Flow field (B, 2, H, W)
        """
        orig_size = source.shape[-2:]
        
        if self.method == "flowformer++" and self.flowformer is not None:
            # Upscale for better FlowFormer++ estimation
            source_up, scale = self.upscale_image(source, self.min_size)
            target_up, _ = self.upscale_image(target, self.min_size)
            
            # Compute flow on upscaled images
            flow = self.flowformer.compute_flow(source_up, target_up)
            
            # Downscale flow back to original size
            flow = self.downscale_flow(flow, scale, orig_size)
            
            return flow
        else:
            return self._correlation_flow(source, target)
    
    def _correlation_flow(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        search_range: int = 4,
    ) -> torch.Tensor:
        """
        Simple correlation-based optical flow for small images.
        
        Args:
            source: Source frame (B, C, H, W)
            target: Target frame (B, C, H, W)
            search_range: Maximum displacement in pixels
            
        Returns:
            Flow field (B, 2, H, W)
        """
        B, C, H, W = source.shape
        
        # Convert to grayscale
        source_gray = source.mean(dim=1, keepdim=True)
        target_gray = target.mean(dim=1, keepdim=True)
        
        # Initialize flow
        flow = torch.zeros(B, 2, H, W, device=source.device)
        
        # Pad target
        pad = search_range
        target_padded = F.pad(target_gray, (pad, pad, pad, pad), mode='replicate')
        
        # Search for best match
        best_corr = torch.full((B, 1, H, W), -float('inf'), device=source.device)
        
        for dy in range(-search_range, search_range + 1):
            for dx in range(-search_range, search_range + 1):
                y_start = pad + dy
                y_end = pad + dy + H
                x_start = pad + dx
                x_end = pad + dx + W
                
                target_shifted = target_padded[:, :, y_start:y_end, x_start:x_end]
                corr = -torch.abs(source_gray - target_shifted).mean(dim=1, keepdim=True)
                
                better = corr > best_corr
                best_corr = torch.where(better, corr, best_corr)
                flow[:, 0:1] = torch.where(better, torch.full_like(corr, dx), flow[:, 0:1])
                flow[:, 1:2] = torch.where(better, torch.full_like(corr, dy), flow[:, 1:2])
        
        return flow
    
    def compute_bidirectional_flow(
        self,
        center_frame: torch.Tensor,
        neighbor_frames: List[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Compute bidirectional optical flow between center and neighbors.
        
        Args:
            center_frame: Center frame (B, C, H, W)
            neighbor_frames: List of neighboring frames
            
        Returns:
            Tuple of (forward_flows, backward_flows)
        """
        forward_flows = []
        backward_flows = []
        
        for neighbor in neighbor_frames:
            # Forward: center -> neighbor
            forward = self.compute_flow(center_frame, neighbor)
            forward_flows.append(forward)
            
            # Backward: neighbor -> center
            backward = self.compute_flow(neighbor, center_frame)
            backward_flows.append(backward)
        
        return forward_flows, backward_flows


def warp_flow(image: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """
    Warp an image using optical flow.
    
    Args:
        image: Image to warp (B, C, H, W)
        flow: Optical flow (B, 2, H, W), flow[0]=dx, flow[1]=dy
        
    Returns:
        Warped image (B, C, H, W)
    """
    B, C, H, W = image.shape
    
    # Create base grid
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=image.device, dtype=torch.float32),
        torch.arange(W, device=image.device, dtype=torch.float32),
        indexing='ij'
    )
    grid_x = grid_x.unsqueeze(0).expand(B, -1, -1)
    grid_y = grid_y.unsqueeze(0).expand(B, -1, -1)
    
    # Add flow to grid
    new_x = grid_x + flow[:, 0, :, :]
    new_y = grid_y + flow[:, 1, :, :]
    
    # Normalize to [-1, 1]
    new_x = 2.0 * new_x / max(W - 1, 1) - 1.0
    new_y = 2.0 * new_y / max(H - 1, 1) - 1.0
    
    # Stack and warp
    grid = torch.stack([new_x, new_y], dim=-1)
    warped = F.grid_sample(image, grid, mode='bilinear', padding_mode='border', align_corners=True)
    
    return warped
