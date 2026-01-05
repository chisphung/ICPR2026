"""
Dataset class for license plate sequences.
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class LicensePlateDataset(Dataset):
    """Dataset for loading license plate image sequences."""
    
    def __init__(
        self,
        data_dir: Path,
        scenarios: Optional[List[str]] = None,
        plate_types: Optional[List[str]] = None,
        transform=None,
        load_hr: bool = True,
    ):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Root directory containing train/test data
            scenarios: List of scenarios to include (e.g., ['Scenario-A', 'Scenario-B'])
            plate_types: List of plate types to include (e.g., ['Brazilian', 'Mercosur'])
            transform: Optional transforms to apply to images
            load_hr: Whether to load high-resolution ground truth images
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.load_hr = load_hr
        
        # Default to all scenarios and plate types
        if scenarios is None:
            scenarios = ['Scenario-A', 'Scenario-B']
        if plate_types is None:
            plate_types = ['Brazilian', 'Mercosur']
        
        # Collect all track paths
        self.tracks = []
        for scenario in scenarios:
            for plate_type in plate_types:
                scenario_path = self.data_dir / scenario / plate_type
                if scenario_path.exists():
                    tracks = sorted(scenario_path.glob("track_*"))
                    self.tracks.extend(tracks)
        
        print(f"Found {len(self.tracks)} tracks")
    
    def __len__(self) -> int:
        return len(self.tracks)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single sample.
        
        Returns:
            Dict containing:
                - lr_frames: List of low-resolution frames (numpy arrays)
                - hr_frames: List of high-resolution frames (if load_hr=True)
                - plate_text: Ground truth plate text
                - corners: License plate corner coordinates
                - track_path: Path to the track directory
        """
        track_path = self.tracks[idx]
        
        # Load annotations
        annotations_path = track_path / "annotations.json"
        with open(annotations_path, 'r') as f:
            annotations = json.load(f)
        
        # Load LR frames
        lr_frames = []
        for i in range(1, 6):  # lr-001.png to lr-005.png
            lr_path = track_path / f"lr-{i:03d}.png"
            img = Image.open(lr_path).convert('RGB')
            lr_frames.append(np.array(img))
        
        # Load HR frames if requested
        hr_frames = []
        if self.load_hr:
            for i in range(1, 6):  # hr-001.png to hr-005.png
                hr_path = track_path / f"hr-{i:03d}.png"
                img = Image.open(hr_path).convert('RGB')
                hr_frames.append(np.array(img))
        
        # Apply transforms
        if self.transform:
            lr_frames = [self.transform(f) for f in lr_frames]
            if self.load_hr:
                hr_frames = [self.transform(f) for f in hr_frames]
        
        return {
            'lr_frames': lr_frames,
            'hr_frames': hr_frames if self.load_hr else None,
            'plate_text': annotations.get('plate_text', ''),
            'plate_layout': annotations.get('plate_layout', ''),
            'corners': annotations.get('corners', {}),
            'track_path': str(track_path),
        }


def load_single_track(track_path: Path) -> Dict:
    """
    Load a single track for inference.
    
    Args:
        track_path: Path to track directory
        
    Returns:
        Dict with lr_frames, hr_frames, plate_text, corners
    """
    track_path = Path(track_path)
    
    # Load annotations
    annotations_path = track_path / "annotations.json"
    with open(annotations_path, 'r') as f:
        annotations = json.load(f)
    
    # Load LR frames
    lr_frames = []
    for i in range(1, 6):
        lr_path = track_path / f"lr-{i:03d}.png"
        img = Image.open(lr_path).convert('RGB')
        lr_frames.append(np.array(img))
    
    # Load HR frames
    hr_frames = []
    for i in range(1, 6):
        hr_path = track_path / f"hr-{i:03d}.png"
        if hr_path.exists():
            img = Image.open(hr_path).convert('RGB')
            hr_frames.append(np.array(img))
    
    return {
        'lr_frames': lr_frames,
        'hr_frames': hr_frames,
        'plate_text': annotations.get('plate_text', ''),
        'plate_layout': annotations.get('plate_layout', ''),
        'corners': annotations.get('corners', {}),
        'track_path': str(track_path),
    }


def frames_to_tensor(frames: List[np.ndarray], device: str = 'cuda') -> torch.Tensor:
    """
    Convert list of numpy frames to tensor.
    
    Args:
        frames: List of numpy arrays (H, W, C) in range [0, 255]
        device: Target device
        
    Returns:
        Tensor of shape (N, C, H, W) in range [0, 1]
    """
    # Find max dimensions to handle varying sizes
    max_h = max(f.shape[0] for f in frames)
    max_w = max(f.shape[1] for f in frames)
    
    tensors = []
    for frame in frames:
        # Resize if needed to match max dimensions
        if frame.shape[0] != max_h or frame.shape[1] != max_w:
            from PIL import Image
            img = Image.fromarray(frame)
            img = img.resize((max_w, max_h), Image.Resampling.BILINEAR)
            frame = np.array(img)
        
        # Convert to float and normalize
        frame = frame.astype(np.float32) / 255.0
        # Convert HWC to CHW
        frame = np.transpose(frame, (2, 0, 1))
        tensors.append(torch.from_numpy(frame))
    
    return torch.stack(tensors).to(device)


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert tensor to numpy image.
    
    Args:
        tensor: Tensor of shape (C, H, W) or (H, W, C) in range [0, 1]
        
    Returns:
        Numpy array of shape (H, W, C) in range [0, 255]
    """
    if tensor.dim() == 3 and tensor.shape[0] in [1, 3]:
        # CHW format, convert to HWC
        tensor = tensor.permute(1, 2, 0)
    
    img = tensor.detach().cpu().numpy()
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img
