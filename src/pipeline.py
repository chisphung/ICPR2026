"""
MF-LPR² End-to-End Pipeline.

Combines all modules for complete license plate restoration and recognition.
"""
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

from .config import MFLPRConfig, DEFAULT_CONFIG
from .data.dataset import load_single_track, frames_to_tensor, tensor_to_numpy
from .models.optical_flow import OpticalFlowEstimator
from .models.temporal_filter import TemporalFilter
from .models.spatial_refiner import SpatialRefiner, corners_from_annotations
from .models.gsr4 import GSR4Aggregator
from .models.recognizer import LicensePlateRecognizer


class MFLPRPipeline(nn.Module):
    """
    Complete MF-LPR² pipeline for license plate restoration and recognition.
    
    Pipeline stages:
    1. Load frame sequence
    2. Estimate optical flows from center to neighbors
    3. Apply temporal filtering to reject unreliable frames
    4. Apply spatial refinement using homography
    5. Aggregate frames using GSR4
    6. Recognize text using MGP-STR
    """
    
    def __init__(self, config: Optional[MFLPRConfig] = None):
        """
        Initialize the pipeline.
        
        Args:
            config: Configuration object (uses DEFAULT_CONFIG if None)
        """
        super().__init__()
        self.config = config or DEFAULT_CONFIG
        
        # Initialize modules
        self.flow_estimator = OpticalFlowEstimator(
            method="flowformer++",
            weights_path=self.config.flowformer_weights,
            device=self.config.device,
        )
        
        self.temporal_filter = TemporalFilter(
            theta_temp=self.config.theta_temp,
        )
        
        self.spatial_refiner = SpatialRefiner(
            theta_spatial=self.config.theta_spatial,
        )
        
        self.aggregator = GSR4Aggregator(
            k_neighbors=self.config.gsr4_k_neighbors,
            upscale_factor=2,  # LR to HR upscaling
        )
        
        self.recognizer = LicensePlateRecognizer(
            model_name=self.config.mgp_str_model,
            device=self.config.device,
        )
    
    def restore_frames(
        self,
        lr_frames: torch.Tensor,
        corners: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Restore a high-quality frame from low-resolution sequence.
        
        Args:
            lr_frames: Low-resolution frames (N, C, H, W) where N=5
            corners: Optional corner annotations for spatial refinement
            
        Returns:
            Tuple of (restored_frame, info_dict)
        """
        N, C, H, W = lr_frames.shape
        center_idx = self.config.center_frame_idx
        
        # Extract center and neighbor frames
        center_frame = lr_frames[center_idx].unsqueeze(0)  # (1, C, H, W)
        neighbor_indices = [i for i in range(N) if i != center_idx]
        neighbor_frames = [lr_frames[i].unsqueeze(0) for i in neighbor_indices]
        
        # Step 1: Estimate optical flows
        forward_flows, backward_flows = self.flow_estimator.compute_bidirectional_flow(
            center_frame, neighbor_frames
        )
        
        # Step 2: Temporal filtering
        valid_frames, valid_flows, validity_mask = self.temporal_filter.filter_frames(
            neighbor_frames, forward_flows, backward_flows
        )
        
        # Step 3: Spatial refinement (if corners provided)
        refined_flows = []
        if corners is not None:
            center_corners = self._get_corners(corners, center_idx)
            for i, (frame, flow) in enumerate(zip(valid_frames, valid_flows)):
                orig_idx = neighbor_indices[i] if i < len(neighbor_indices) else i
                neighbor_corners = self._get_corners(corners, orig_idx)
                
                if center_corners is not None and neighbor_corners is not None:
                    # Move corners to same device as flow
                    device = flow.device
                    refined, _ = self.spatial_refiner.refine_flow(
                        flow, 
                        neighbor_corners.unsqueeze(0).to(device),
                        center_corners.unsqueeze(0).to(device),
                    )
                    refined_flows.append(refined)
                else:
                    refined_flows.append(flow)
        else:
            refined_flows = valid_flows
        
        # Step 4: GSR4 Aggregation
        if len(valid_frames) > 0:
            restored = self.aggregator.forward(
                center_frame, valid_frames, refined_flows, upscale=True
            )
        else:
            # No valid neighbors, just upscale center
            restored = self.aggregator.upscale(center_frame)
        
        info = {
            'num_valid_frames': len(valid_frames),
            'validity_mask': validity_mask,
        }
        
        return restored, info
    
    def _get_corners(self, corners: Dict, frame_idx: int) -> Optional[torch.Tensor]:
        """Extract corners for a specific frame index."""
        frame_name = f"lr-{frame_idx + 1:03d}.png"
        if frame_name in corners:
            c = corners[frame_name]
            return torch.tensor([
                c['top-left'],
                c['top-right'],
                c['bottom-right'],
                c['bottom-left'],
            ], dtype=torch.float32)
        return None
    
    def recognize(self, image: torch.Tensor) -> List[str]:
        """
        Recognize text from restored image.
        
        Args:
            image: Restored image (B, C, H, W)
            
        Returns:
            List of recognized text strings
        """
        return self.recognizer.recognize(image)
    
    def forward(
        self,
        lr_frames: torch.Tensor,
        corners: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, List[str], Dict]:
        """
        Full pipeline: restore and recognize.
        
        Args:
            lr_frames: Low-resolution frames (N, C, H, W)
            corners: Optional corner annotations
            
        Returns:
            Tuple of (restored_image, predicted_texts, info_dict)
        """
        # Restore
        restored, info = self.restore_frames(lr_frames, corners)
        
        # Recognize
        predictions = self.recognize(restored)
        
        return restored, predictions, info


def run_inference(
    track_path: Path,
    config: Optional[MFLPRConfig] = None,
    save_output: bool = True,
) -> Dict:
    """
    Run inference on a single track.
    
    Args:
        track_path: Path to track directory
        config: Configuration object
        save_output: Whether to save the restored image
        
    Returns:
        Dict with results including restored image and prediction
    """
    config = config or DEFAULT_CONFIG
    track_path = Path(track_path)
    
    # Load data
    data = load_single_track(track_path)
    
    # Convert to tensors
    lr_frames = frames_to_tensor(data['lr_frames'], device=config.device)
    
    # Initialize pipeline
    pipeline = MFLPRPipeline(config)
    
    # Run inference
    with torch.no_grad():
        restored, predictions, info = pipeline(lr_frames, data['corners'])
    
    # Convert restored to numpy
    restored_np = tensor_to_numpy(restored[0])
    
    # Prepare results
    results = {
        'track_path': str(track_path),
        'ground_truth': data['plate_text'],
        'prediction': predictions[0] if predictions else '',
        'restored_image': restored_np,
        'num_valid_frames': info['num_valid_frames'],
        'validity_mask': info['validity_mask'],
    }
    
    # Save if requested
    if save_output:
        from PIL import Image
        output_dir = config.output_dir / track_path.name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        Image.fromarray(restored_np).save(output_dir / 'restored.png')
        
        # Save result summary
        with open(output_dir / 'result.txt', 'w') as f:
            f.write(f"Ground Truth: {data['plate_text']}\n")
            f.write(f"Prediction: {predictions[0] if predictions else 'N/A'}\n")
            f.write(f"Valid Frames: {info['num_valid_frames']}/{len(data['lr_frames'])-1}\n")
    
    return results
