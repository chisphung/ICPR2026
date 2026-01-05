"""
Temporal Filtering Module.

Implements temporal consistency check to filter out frames with
unreliable optical flow estimates (Equations 3-4 in the paper).
"""
import torch
import torch.nn as nn
from typing import List, Tuple

from .optical_flow import warp_flow


class TemporalFilter(nn.Module):
    """
    Temporal filtering based on bidirectional optical flow consistency.
    
    Frames are rejected if the forward-backward flow consistency error
    exceeds the temporal threshold.
    """
    
    def __init__(self, theta_temp: float = 10.0):
        """
        Initialize temporal filter.
        
        Args:
            theta_temp: Temporal consistency threshold (default: 10.0 from paper)
        """
        super().__init__()
        self.theta_temp = theta_temp
    
    def compute_consistency_error(
        self,
        forward_flow: torch.Tensor,
        backward_flow: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute bidirectional flow consistency error.
        
        Equation 3: Diff(f, f') = ||f(i,j) + f'(i+u, j+v)|| - median
        
        Args:
            forward_flow: Forward flow (B, 2, H, W)
            backward_flow: Backward flow (B, 2, H, W)
            
        Returns:
            Consistency error per pixel (B, 1, H, W)
        """
        # Warp backward flow using forward flow
        warped_backward = warp_flow(backward_flow, forward_flow)
        
        # Compute flow sum: f + f'(i+u, j+v)
        flow_sum = forward_flow + warped_backward
        
        # Compute magnitude of the sum
        error_magnitude = torch.norm(flow_sum, dim=1, keepdim=True)
        
        # Compute median for normalization
        B = error_magnitude.shape[0]
        median = error_magnitude.view(B, -1).median(dim=1, keepdim=True)[0]
        median = median.view(B, 1, 1, 1)
        
        # Diff = magnitude - median
        diff = error_magnitude - median
        
        return diff
    
    def check_frame_validity(
        self,
        forward_flow: torch.Tensor,
        backward_flow: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Check if a frame should be included based on temporal consistency.
        
        Equation 4: max(Diff) - median(Diff) > θ_temp → reject
        
        Args:
            forward_flow: Forward flow (B, 2, H, W)
            backward_flow: Backward flow (B, 2, H, W)
            
        Returns:
            Tuple of (is_valid, error_map)
            - is_valid: Boolean tensor (B,) indicating if frame is valid
            - error_map: Per-pixel error (B, 1, H, W)
        """
        diff = self.compute_consistency_error(forward_flow, backward_flow)
        
        B = diff.shape[0]
        diff_flat = diff.view(B, -1)
        
        # max(Diff) - median(Diff)
        max_diff = diff_flat.max(dim=1)[0]
        median_diff = diff_flat.median(dim=1)[0]
        
        score = max_diff - median_diff
        
        # Frame is valid if score <= theta_temp
        is_valid = score <= self.theta_temp
        
        return is_valid, diff
    
    def filter_frames(
        self,
        neighbor_frames: List[torch.Tensor],
        forward_flows: List[torch.Tensor],
        backward_flows: List[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[bool]]:
        """
        Filter frames based on temporal consistency.
        
        Args:
            neighbor_frames: List of neighboring frames
            forward_flows: List of forward flows (center → neighbor)
            backward_flows: List of backward flows (neighbor → center)
            
        Returns:
            Tuple of (valid_frames, valid_flows, validity_mask)
        """
        valid_frames = []
        valid_flows = []
        validity_mask = []
        
        for frame, fwd_flow, bwd_flow in zip(neighbor_frames, forward_flows, backward_flows):
            is_valid, _ = self.check_frame_validity(fwd_flow, bwd_flow)
            
            # For batch processing, check if majority of batch is valid
            batch_valid = is_valid.float().mean() > 0.5
            
            validity_mask.append(batch_valid.item())
            
            if batch_valid:
                valid_frames.append(frame)
                valid_flows.append(bwd_flow)  # Use backward flow for warping to center
        
        return valid_frames, valid_flows, validity_mask
