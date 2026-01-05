"""
FlowFormer++ Optical Flow Estimation Wrapper.

This module provides a clean interface to FlowFormer++ for optical flow estimation.
"""
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional


class FlowFormerPPWrapper(nn.Module):
    """
    Wrapper for FlowFormer++ optical flow estimation.
    """
    
    def __init__(
        self,
        weights_path: Path,
        device: str = "cuda",
        flowformer_dir: Optional[Path] = None,
    ):
        """
        Initialize FlowFormer++ model.
        
        Args:
            weights_path: Path to pretrained weights (.pth file)
            device: Target device
            flowformer_dir: Path to FlowFormer++ repository
        """
        super().__init__()
        self.device = device
        self.weights_path = Path(weights_path).resolve()
        
        # Find FlowFormer++ directory
        if flowformer_dir is None:
            # Look relative to this file's location
            src_dir = Path(__file__).parent.parent.parent
            flowformer_dir = src_dir / "flowformer"
        
        self.flowformer_dir = Path(flowformer_dir).resolve()
        self.model = None
        self._loaded = False
        
    def load_model(self):
        """Load FlowFormer++ model with pretrained weights."""
        if self._loaded:
            return
        
        # Save current directory and sys.path
        original_cwd = os.getcwd()
        original_path = sys.path.copy()
        
        try:
            # Change to FlowFormer++ directory for proper imports
            os.chdir(self.flowformer_dir)
            sys.path.insert(0, str(self.flowformer_dir))
            sys.path.insert(0, str(self.flowformer_dir / "core"))
            
            # Import FlowFormer++ components
            from core.FlowFormer import build_flowformer
            from configs.things import get_cfg
            
            # Get config
            cfg = get_cfg()
            
            # Build model
            self.model = build_flowformer(cfg)
            
            # Load weights
            state_dict = torch.load(self.weights_path, map_location=self.device)
            
            # Handle DataParallel wrapped state dict
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            
            self.model.load_state_dict(new_state_dict, strict=False)
            self.model = self.model.to(self.device)
            self.model.eval()
            self._loaded = True
            print(f"✓ Loaded FlowFormer++ from {self.weights_path}")
            
        except Exception as e:
            print(f"Error loading FlowFormer++: {e}")
            import traceback
            traceback.print_exc()
            self._loaded = False
            
        finally:
            # Restore original directory and path
            os.chdir(original_cwd)
            sys.path = original_path
    
    def pad_to_multiple(self, img: torch.Tensor, multiple: int = 8) -> Tuple[torch.Tensor, Tuple]:
        """Pad image dimensions to be divisible by multiple."""
        H, W = img.shape[-2:]
        pad_h = (multiple - H % multiple) % multiple
        pad_w = (multiple - W % multiple) % multiple
        
        if pad_h > 0 or pad_w > 0:
            img = F.pad(img, [0, pad_w, 0, pad_h], mode='replicate')
        
        return img, (pad_h, pad_w)
    
    def unpad(self, flow: torch.Tensor, pad: Tuple) -> torch.Tensor:
        """Remove padding from flow."""
        pad_h, pad_w = pad
        if pad_h > 0:
            flow = flow[:, :, :-pad_h, :]
        if pad_w > 0:
            flow = flow[:, :, :, :-pad_w]
        return flow
    
    @torch.no_grad()
    def compute_flow(
        self,
        image1: torch.Tensor,
        image2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute optical flow from image1 to image2.
        
        Args:
            image1: Source image (B, C, H, W) normalized to [0, 1]
            image2: Target image (B, C, H, W) normalized to [0, 1]
            
        Returns:
            Optical flow (B, 2, H, W)
        """
        if not self._loaded:
            self.load_model()
        
        if self.model is None:
            # Fallback to correlation-based flow
            return self._correlation_flow(image1, image2)
        
        # FlowFormer++ expects images in [0, 255] range
        img1 = image1 * 255.0
        img2 = image2 * 255.0
        
        # Pad to multiple of 8
        img1_padded, pad1 = self.pad_to_multiple(img1, 8)
        img2_padded, _ = self.pad_to_multiple(img2, 8)
        
        # Run FlowFormer++
        flow_predictions = self.model(img1_padded, img2_padded)
        
        # Get the final flow prediction (first element of tuple)
        if isinstance(flow_predictions, (list, tuple)):
            flow = flow_predictions[0]
        else:
            flow = flow_predictions
        
        # Remove padding
        flow = self.unpad(flow, pad1)
        
        return flow
    
    def _correlation_flow(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        search_range: int = 4,
    ) -> torch.Tensor:
        """Fallback correlation-based optical flow for small images."""
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
