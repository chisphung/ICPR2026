"""
License Plate Recognition Module using MGP-STR.

Integrates the MGP-STR model from HuggingFace for scene text recognition.
"""
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from typing import List, Optional, Union


class LicensePlateRecognizer(nn.Module):
    """
    License plate text recognition using MGP-STR.
    
    MGP-STR is a scene text recognition model that combines
    visual features with multi-granularity prediction.
    """
    
    def __init__(
        self,
        model_name: str = "alibaba-damo/mgp-str-base",
        device: str = "cuda",
    ):
        """
        Initialize the recognizer.
        
        Args:
            model_name: HuggingFace model name for MGP-STR
            device: Target device
        """
        super().__init__()
        self.model_name = model_name
        self.device = device
        self.model = None
        self.processor = None
        self._loaded = False
    
    def load_model(self):
        """Load the MGP-STR model and processor."""
        if self._loaded:
            return
        
        try:
            from transformers import MgpstrProcessor, MgpstrForSceneTextRecognition
            
            self.processor = MgpstrProcessor.from_pretrained(self.model_name)
            self.model = MgpstrForSceneTextRecognition.from_pretrained(self.model_name)
            self.model = self.model.to(self.device)
            self.model.eval()
            self._loaded = True
            print(f"Loaded MGP-STR model: {self.model_name}")
            
        except ImportError:
            print("Warning: transformers not installed. Using fallback mode.")
            self._loaded = False
        except Exception as e:
            print(f"Error loading MGP-STR: {e}")
            self._loaded = False
    
    def preprocess(
        self,
        images: Union[np.ndarray, List[np.ndarray], torch.Tensor],
    ) -> torch.Tensor:
        """
        Preprocess images for MGP-STR.
        
        Args:
            images: Input images (numpy array or tensor)
            
        Returns:
            Preprocessed pixel values tensor
        """
        if not self._loaded:
            self.load_model()
        
        if self.processor is None:
            raise RuntimeError("Model not loaded")
        
        # Convert tensor to PIL Image
        if isinstance(images, torch.Tensor):
            if images.dim() == 4:
                images = [self._tensor_to_pil(img) for img in images]
            else:
                images = [self._tensor_to_pil(images)]
        elif isinstance(images, np.ndarray):
            if images.ndim == 4:
                images = [Image.fromarray(img) for img in images]
            else:
                images = [Image.fromarray(images)]
        elif isinstance(images, list):
            processed = []
            for img in images:
                if isinstance(img, np.ndarray):
                    processed.append(Image.fromarray(img))
                elif isinstance(img, torch.Tensor):
                    processed.append(self._tensor_to_pil(img))
                else:
                    processed.append(img)
            images = processed
        
        # Convert to RGB
        images = [img.convert("RGB") if isinstance(img, Image.Image) else img for img in images]
        
        # Process through MGP-STR processor
        pixel_values = self.processor(images=images, return_tensors="pt").pixel_values
        
        return pixel_values.to(self.device)
    
    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """Convert tensor to PIL Image."""
        if tensor.dim() == 3:
            if tensor.shape[0] in [1, 3]:
                # CHW format
                tensor = tensor.permute(1, 2, 0)
        
        # Convert to numpy and scale to [0, 255]
        img = tensor.detach().cpu().numpy()
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
        
        return Image.fromarray(img)
    
    @torch.no_grad()
    def recognize(
        self,
        images: Union[np.ndarray, List[np.ndarray], torch.Tensor],
    ) -> List[str]:
        """
        Recognize text from license plate images.
        
        Args:
            images: Input images (B, C, H, W) or list of images
            
        Returns:
            List of recognized text strings
        """
        if not self._loaded:
            self.load_model()
        
        if self.model is None:
            # Fallback: return empty strings
            if isinstance(images, (list, tuple)):
                return [""] * len(images)
            return [""]
        
        # Preprocess
        pixel_values = self.preprocess(images)
        
        # Inference
        outputs = self.model(pixel_values)
        
        # Decode predictions
        results = self.processor.batch_decode(outputs.logits)
        
        # Extract generated text - results is a dict like {'generated_text': [...], 'scores': [...], ...}
        if isinstance(results, dict):
            predictions = results.get("generated_text", [""])
        elif isinstance(results, list):
            predictions = results
        else:
            predictions = [str(results)]
        
        return predictions
    
    def forward(
        self,
        images: torch.Tensor,
    ) -> List[str]:
        """
        Forward pass for recognition.
        
        Args:
            images: Input tensor (B, C, H, W)
            
        Returns:
            List of recognized text strings
        """
        return self.recognize(images)


class SimpleCRNNRecognizer(nn.Module):
    """
    Simple CRNN-based recognizer as fallback.
    
    Used when MGP-STR is not available.
    """
    
    # License plate character set
    CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    
    def __init__(self, num_classes: int = None, hidden_size: int = 256):
        super().__init__()
        
        if num_classes is None:
            num_classes = len(self.CHARS) + 1  # +1 for blank
        
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        self.rnn = nn.LSTM(128, hidden_size, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Logits (B, T, num_classes)
        """
        # CNN features
        features = self.cnn(x)  # (B, C, H', W')
        
        # Reshape for RNN: (B, W', C*H')
        B, C, H, W = features.shape
        features = features.permute(0, 3, 1, 2).contiguous()  # (B, W, C, H)
        features = features.view(B, W, -1)  # (B, W, C*H)
        
        # RNN
        rnn_out, _ = self.rnn(features)  # (B, W, hidden*2)
        
        # FC
        output = self.fc(rnn_out)  # (B, W, num_classes)
        
        return output
    
    def decode(self, logits: torch.Tensor) -> List[str]:
        """
        Decode logits to text using CTC greedy decoding.
        
        Args:
            logits: Output logits (B, T, num_classes)
            
        Returns:
            List of decoded strings
        """
        # Greedy decoding
        preds = logits.argmax(dim=-1)  # (B, T)
        
        results = []
        for pred in preds:
            chars = []
            prev = -1
            for idx in pred.tolist():
                if idx != prev and idx != 0:  # 0 is blank
                    if idx <= len(self.CHARS):
                        chars.append(self.CHARS[idx - 1])
                prev = idx
            results.append("".join(chars))
        
        return results
