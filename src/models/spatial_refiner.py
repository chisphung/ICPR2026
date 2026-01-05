"""
Spatial Refinement Module.

Implements spatial consistency refinement using homography estimation
to correct local optical flow errors (Equation 5 in the paper).
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


class SpatialRefiner(nn.Module):
    """
    Spatial refinement using homography-based flow correction.
    
    Estimates a homography between frame corners and uses it to
    correct local optical flow errors in non-planar regions.
    """
    
    def __init__(self, theta_spatial: float = 20.0):
        """
        Initialize spatial refiner.
        
        Args:
            theta_spatial: Spatial consistency threshold (default: 20.0 from paper)
        """
        super().__init__()
        self.theta_spatial = theta_spatial
    
    def estimate_homography(
        self,
        src_corners: torch.Tensor,
        dst_corners: torch.Tensor,
    ) -> torch.Tensor:
        """
        Estimate homography matrix from corner correspondences.
        
        Args:
            src_corners: Source corners (B, 4, 2)
            dst_corners: Destination corners (B, 4, 2)
            
        Returns:
            Homography matrix (B, 3, 3)
        """
        B = src_corners.shape[0]
        device = src_corners.device
        
        # Build the DLT matrix
        H_list = []
        
        for b in range(B):
            src = src_corners[b]  # (4, 2)
            dst = dst_corners[b]  # (4, 2)
            
            # Build matrix A for DLT
            A = []
            for i in range(4):
                x, y = src[i]
                u, v = dst[i]
                A.append([-x, -y, -1, 0, 0, 0, u*x, u*y, u])
                A.append([0, 0, 0, -x, -y, -1, v*x, v*y, v])
            
            A = torch.tensor(A, device=device, dtype=torch.float32)
            
            # SVD to find homography
            try:
                _, _, Vh = torch.linalg.svd(A)
                h = Vh[-1, :]
                H = h.reshape(3, 3)
                H = H / H[2, 2]  # Normalize
            except:
                # Fallback to identity
                H = torch.eye(3, device=device)
            
            H_list.append(H)
        
        return torch.stack(H_list)
    
    def apply_homography(
        self,
        points: torch.Tensor,
        H: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply homography to points.
        
        Args:
            points: Points (B, N, 2)
            H: Homography matrix (B, 3, 3)
            
        Returns:
            Transformed points (B, N, 2)
        """
        B, N, _ = points.shape
        
        # Convert to homogeneous coordinates
        ones = torch.ones(B, N, 1, device=points.device)
        points_h = torch.cat([points, ones], dim=-1)  # (B, N, 3)
        
        # Apply homography
        # H @ points^T -> (B, 3, N) then transpose -> (B, N, 3)
        transformed = torch.bmm(H, points_h.transpose(1, 2)).transpose(1, 2)
        
        # Convert from homogeneous
        transformed = transformed[:, :, :2] / (transformed[:, :, 2:3] + 1e-8)
        
        return transformed
    
    def compute_homography_flow(
        self,
        H: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """
        Compute dense optical flow from homography.
        
        Args:
            H: Homography matrix (B, 3, 3)
            height: Image height
            width: Image width
            
        Returns:
            Flow field (B, 2, H, W)
        """
        B = H.shape[0]
        device = H.device
        
        # Create grid of points
        y, x = torch.meshgrid(
            torch.arange(height, device=device, dtype=torch.float32),
            torch.arange(width, device=device, dtype=torch.float32),
            indexing='ij'
        )
        
        # Flatten and stack
        points = torch.stack([x.flatten(), y.flatten()], dim=-1)  # (H*W, 2)
        points = points.unsqueeze(0).expand(B, -1, -1)  # (B, H*W, 2)
        
        # Apply homography
        transformed = self.apply_homography(points, H)  # (B, H*W, 2)
        
        # Compute flow as displacement
        flow = transformed - points  # (B, H*W, 2)
        
        # Reshape to (B, H, W, 2) then permute to (B, 2, H, W)
        flow = flow.view(B, height, width, 2).permute(0, 3, 1, 2)
        
        return flow
    
    def refine_flow(
        self,
        optical_flow: torch.Tensor,
        src_corners: torch.Tensor,
        dst_corners: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Refine optical flow using homography constraint.
        
        Equation 5: Detect outliers where max(Diff) - median(Diff) > θ_spatial
        
        Args:
            optical_flow: Estimated optical flow (B, 2, H, W)
            src_corners: Source license plate corners (B, 4, 2)
            dst_corners: Destination license plate corners (B, 4, 2)
            
        Returns:
            Tuple of (refined_flow, outlier_mask)
        """
        B, _, H, W = optical_flow.shape
        
        # Estimate homography from corners
        H_matrix = self.estimate_homography(src_corners, dst_corners)
        
        # Compute homography-based flow
        homography_flow = self.compute_homography_flow(H_matrix, H, W)
        
        # Compute difference between optical flow and homography flow
        diff = torch.norm(optical_flow - homography_flow, dim=1, keepdim=True)
        
        # Compute outlier score
        diff_flat = diff.view(B, -1)
        max_diff = diff_flat.max(dim=1, keepdim=True)[0]
        median_diff = diff_flat.median(dim=1, keepdim=True)[0]
        
        score = max_diff - median_diff  # (B, 1)
        
        # Pixels are outliers if local error is high
        pixel_threshold = median_diff.view(B, 1, 1, 1) + self.theta_spatial
        outlier_mask = diff > pixel_threshold
        
        # Refine: replace outlier pixels with homography flow
        refined_flow = torch.where(outlier_mask.expand(-1, 2, -1, -1), 
                                   homography_flow, 
                                   optical_flow)
        
        return refined_flow, outlier_mask


def corners_from_annotations(corners_dict: dict, frame_name: str) -> torch.Tensor:
    """
    Extract corner coordinates from annotation dictionary.
    
    Args:
        corners_dict: Corner annotations from JSON
        frame_name: Frame filename (e.g., 'lr-001.png')
        
    Returns:
        Corner tensor (4, 2) in order [TL, TR, BR, BL]
    """
    if frame_name not in corners_dict:
        return None
    
    c = corners_dict[frame_name]
    corners = torch.tensor([
        c['top-left'],
        c['top-right'],
        c['bottom-right'],
        c['bottom-left'],
    ], dtype=torch.float32)
    
    return corners
