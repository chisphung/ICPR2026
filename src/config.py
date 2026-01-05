"""
Configuration for MF-LPR² framework.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class MFLPRConfig:
    """Configuration for MF-LPR² pipeline."""
    
    # Paths
    data_dir: Path = field(default_factory=lambda: Path("dataset/train"))
    # flowformer_weights: Path = field(default_factory=lambda: Path("flowformerpp_weights/things.pth"))
    flowformer_weights: Path = field(default_factory=lambda: Path("flowformerpp_weights/things_288960.pth"))
    output_dir: Path = field(default_factory=lambda: Path("outputs"))
    
    # Frame settings
    num_frames: int = 5
    center_frame_idx: int = 2  # 0-indexed, middle frame of 5
    
    # Temporal filtering threshold (from paper)
    theta_temp: float = 10.0
    
    # Spatial refinement threshold (from paper)
    theta_spatial: float = 20.0
    
    # GSR4 settings
    gsr4_k_neighbors: int = 1  # k-nearest neighbors per frame
    
    # Recognition settings
    mgp_str_model: str = "alibaba-damo/mgp-str-base"
    
    # Device
    device: str = "cuda"
    
    # Image settings
    lr_height: int = 16
    lr_width: int = 32
    hr_height: int = 32
    hr_width: int = 64
    
    def __post_init__(self):
        """Convert string paths to Path objects."""
        if isinstance(self.data_dir, str):
            self.data_dir = Path(self.data_dir)
        if isinstance(self.flowformer_weights, str):
            self.flowformer_weights = Path(self.flowformer_weights)
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)


# Default configuration
DEFAULT_CONFIG = MFLPRConfig()
