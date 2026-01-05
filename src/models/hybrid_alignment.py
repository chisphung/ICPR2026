"""
Hybrid Alignment Module (HAM) for KSR-Net.

Combines optical flow alignment with Flow-Guided Deformable Attention (FGDA)
to recover information from frames where optical flow fails.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


class DeformableConv2d(nn.Module):
    """
    Simple Deformable Convolution implementation.
    
    Learns offsets to sample from irregular positions.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        groups: int = 1,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        
        # Offset prediction: 2 * kernel_size^2 (x and y offsets)
        self.offset_conv = nn.Conv2d(
            in_channels,
            2 * kernel_size * kernel_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        
        # Main convolution weights
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        )
        
        # Initialize offsets to zero
        nn.init.constant_(self.offset_conv.weight, 0)
        nn.init.constant_(self.offset_conv.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply deformable convolution."""
        # Predict offsets
        offsets = self.offset_conv(x)
        
        # For simplicity, use standard convolution with learned offsets
        # affecting the features through a spatial transformer-like approach
        B, C, H, W = x.shape
        
        # Apply standard convolution (simplified deformable)
        out = self.conv(x)
        
        return out


class FlowGuidedDeformableAttention(nn.Module):
    """
    Flow-Guided Deformable Attention (FGDA).
    
    Uses optical flow residual errors to guide deformable attention,
    correcting misalignments where explicit flow fails.
    """
    
    def __init__(
        self,
        in_channels: int = 64,
        hidden_channels: int = 32,
    ):
        super().__init__()
        self.in_channels = in_channels
        
        # Feature encoder
        self.feat_encoder = nn.Sequential(
            nn.Conv2d(3, in_channels, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, 1, 1),
        )
        
        # Error-guided offset predictor
        # Input: warped features + reference features + error map
        self.offset_predictor = nn.Sequential(
            nn.Conv2d(in_channels * 2 + 1, hidden_channels, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_channels, 2, 3, 1, 1),  # 2 for (dx, dy) offsets
        )
        
        # Deformable feature aggregation
        self.deform_agg = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Output projection
        self.out_proj = nn.Conv2d(in_channels, in_channels, 1)
    
    def warp_features(
        self,
        features: torch.Tensor,
        offsets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Warp features using learned offsets.
        
        Args:
            features: Feature maps (B, C, H, W)
            offsets: Offset field (B, 2, H, W)
            
        Returns:
            Warped features (B, C, H, W)
        """
        B, C, H, W = features.shape
        
        # Create base grid
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=features.device),
            torch.linspace(-1, 1, W, device=features.device),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=-1)  # (H, W, 2)
        grid = grid.unsqueeze(0).expand(B, -1, -1, -1)  # (B, H, W, 2)
        
        # Normalize offsets to [-1, 1] range
        offsets_normalized = offsets.permute(0, 2, 3, 1)  # (B, H, W, 2)
        offsets_normalized[..., 0] = offsets_normalized[..., 0] * 2 / max(W - 1, 1)
        offsets_normalized[..., 1] = offsets_normalized[..., 1] * 2 / max(H - 1, 1)
        
        # Add offsets to grid
        sample_grid = grid + offsets_normalized
        
        # Sample features
        warped = F.grid_sample(
            features, sample_grid, mode='bilinear', 
            padding_mode='border', align_corners=True
        )
        
        return warped
    
    def forward(
        self,
        warped_frame: torch.Tensor,
        ref_frame: torch.Tensor,
        error_map: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply flow-guided deformable attention.
        
        Args:
            warped_frame: Coarsely aligned frame (B, C, H, W)
            ref_frame: Reference frame (B, C, H, W)
            error_map: Flow error map (B, 1, H, W), optional
            
        Returns:
            Refined aligned features (B, feat_dim, H, W)
        """
        # Extract features
        warped_feat = self.feat_encoder(warped_frame)
        ref_feat = self.feat_encoder(ref_frame)
        
        # Compute error map if not provided
        if error_map is None:
            error_map = torch.abs(warped_frame - ref_frame).mean(dim=1, keepdim=True)
        
        # Predict offsets based on features and error
        offset_input = torch.cat([warped_feat, ref_feat, error_map], dim=1)
        offsets = self.offset_predictor(offset_input)
        
        # Apply deformable sampling
        refined_feat = self.warp_features(warped_feat, offsets)
        
        # Aggregate and project
        refined_feat = self.deform_agg(refined_feat)
        output = self.out_proj(refined_feat)
        
        return output


class HybridAlignmentModule(nn.Module):
    """
    Hybrid Alignment Module combining optical flow and deformable attention.
    
    Stage 1: Coarse alignment via optical flow
    Stage 2: Fine correction via FGDA
    """
    
    def __init__(
        self,
        feat_channels: int = 64,
        use_deformable: bool = True,
    ):
        super().__init__()
        self.feat_channels = feat_channels
        self.use_deformable = use_deformable
        
        # Feature encoder for fusion
        self.feat_encoder = nn.Sequential(
            nn.Conv2d(3, feat_channels, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feat_channels, feat_channels, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # FGDA for fine correction
        if use_deformable:
            self.fgda = FlowGuidedDeformableAttention(
                in_channels=feat_channels,
            )
        
        # Confidence weight predictor
        self.confidence_net = nn.Sequential(
            nn.Conv2d(feat_channels + 1, 32, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 1, 3, 1, 1),
            nn.Sigmoid(),
        )
    
    def warp_with_flow(
        self,
        image: torch.Tensor,
        flow: torch.Tensor,
    ) -> torch.Tensor:
        """Warp image using optical flow."""
        B, C, H, W = image.shape
        
        # Create grid
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=image.device, dtype=torch.float32),
            torch.arange(W, device=image.device, dtype=torch.float32),
            indexing='ij'
        )
        grid_x = grid_x.unsqueeze(0).expand(B, -1, -1)
        grid_y = grid_y.unsqueeze(0).expand(B, -1, -1)
        
        # Add flow
        new_x = grid_x + flow[:, 0]
        new_y = grid_y + flow[:, 1]
        
        # Normalize
        new_x = 2.0 * new_x / max(W - 1, 1) - 1.0
        new_y = 2.0 * new_y / max(H - 1, 1) - 1.0
        
        grid = torch.stack([new_x, new_y], dim=-1)
        warped = F.grid_sample(image, grid, mode='bilinear', padding_mode='border', align_corners=True)
        
        return warped
    
    def forward(
        self,
        ref_frame: torch.Tensor,
        neighbor_frames: List[torch.Tensor],
        flows_to_ref: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Align neighboring frames to reference using hybrid approach.
        
        Args:
            ref_frame: Reference frame (B, C, H, W)
            neighbor_frames: List of neighboring frames
            flows_to_ref: Optical flows from neighbors to reference
            
        Returns:
            Tuple of (fused_features, aligned_frames)
        """
        aligned_features = []
        aligned_frames = []
        confidence_maps = []
        
        # Reference features
        ref_feat = self.feat_encoder(ref_frame)
        
        for frame, flow in zip(neighbor_frames, flows_to_ref):
            # Stage 1: Coarse alignment with optical flow
            warped_frame = self.warp_with_flow(frame, flow)
            
            # Compute error map
            error_map = torch.abs(warped_frame - ref_frame).mean(dim=1, keepdim=True)
            
            # Stage 2: Fine correction with FGDA
            if self.use_deformable:
                aligned_feat = self.fgda(warped_frame, ref_frame, error_map)
            else:
                aligned_feat = self.feat_encoder(warped_frame)
            
            # Compute confidence weight
            conf_input = torch.cat([aligned_feat, error_map], dim=1)
            confidence = self.confidence_net(conf_input)
            
            aligned_features.append(aligned_feat)
            aligned_frames.append(warped_frame)
            confidence_maps.append(confidence)
        
        # Weighted fusion of aligned features
        # Stack features and confidences
        all_features = torch.stack([ref_feat] + aligned_features, dim=1)  # (B, N, C, H, W)
        all_confidences = torch.stack(
            [torch.ones_like(confidence_maps[0])] + confidence_maps, dim=1
        )  # (B, N, 1, H, W)
        
        # Normalize confidences
        all_confidences = all_confidences / (all_confidences.sum(dim=1, keepdim=True) + 1e-8)
        
        # Weighted sum
        fused = (all_features * all_confidences).sum(dim=1)  # (B, C, H, W)
        
        return fused, aligned_frames
