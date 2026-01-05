"""
Task-Driven Loss Functions for KSR-Net.

Implements composite loss for license plate restoration:
- Charbonnier Loss: Robust pixel loss
- Gradient Profile Loss: Edge sharpness  
- Semantic STR Loss: Text legibility
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class CharbonnierLoss(nn.Module):
    """
    Charbonnier Loss - robust variant of L1.
    
    More forgiving of outliers than L2, produces sharper edges.
    """
    
    def __init__(self, epsilon: float = 1e-6):
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Charbonnier loss."""
        diff = pred - target
        loss = torch.sqrt(diff ** 2 + self.epsilon ** 2)
        return loss.mean()


class GradientProfileLoss(nn.Module):
    """
    Gradient Profile Loss for edge sharpness.
    
    Computes gradient magnitude and minimizes difference.
    """
    
    def __init__(self):
        super().__init__()
        
        # Sobel kernels for gradient computation
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))
    
    def compute_gradient(self, img: torch.Tensor) -> torch.Tensor:
        """Compute gradient magnitude."""
        # Convert to grayscale if needed
        if img.shape[1] == 3:
            gray = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
        else:
            gray = img
        
        # Move kernels to same device and dtype
        sobel_x = self.sobel_x.to(device=gray.device, dtype=gray.dtype)
        sobel_y = self.sobel_y.to(device=gray.device, dtype=gray.dtype)
        
        # Compute gradients
        grad_x = F.conv2d(gray, sobel_x, padding=1)
        grad_y = F.conv2d(gray, sobel_y, padding=1)
        
        # Magnitude
        magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
        
        return magnitude
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute gradient profile loss."""
        pred_grad = self.compute_gradient(pred)
        target_grad = self.compute_gradient(target)
        
        return F.l1_loss(pred_grad, target_grad)


class SemanticSTRLoss(nn.Module):
    """
    Semantic Scene Text Recognition Loss.
    
    Uses a frozen text recognizer to provide gradients
    that enhance text legibility.
    """
    
    def __init__(self, recognizer: Optional[nn.Module] = None):
        super().__init__()
        self.recognizer = recognizer
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(
        self,
        pred: torch.Tensor,
        target_text: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Compute semantic STR loss.
        
        Note: Requires a differentiable text recognizer.
        For now, returns 0 if no recognizer is provided.
        """
        if self.recognizer is None or target_text is None:
            return torch.tensor(0.0, device=pred.device)
        
        # This would require a differentiable recognizer
        # For now, return placeholder
        return torch.tensor(0.0, device=pred.device)


class KSRNetLoss(nn.Module):
    """
    Combined loss function for KSR-Net.
    
    L_total = λ_pix * L_Char + λ_grad * L_Edge + λ_percep * L_LPIPS + λ_sem * L_STR
    """
    
    def __init__(
        self,
        lambda_pix: float = 1.0,
        lambda_grad: float = 0.5,
        lambda_percep: float = 0.1,
        lambda_sem: float = 0.5,
    ):
        super().__init__()
        
        self.lambda_pix = lambda_pix
        self.lambda_grad = lambda_grad
        self.lambda_percep = lambda_percep
        self.lambda_sem = lambda_sem
        
        self.charbonnier = CharbonnierLoss()
        self.gradient = GradientProfileLoss()
        self.semantic = SemanticSTRLoss()
        
        # LPIPS requires pretrained VGG - simplified version
        self.perceptual = PerceptualLoss()
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        target_text: Optional[str] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute total loss.
        
        Returns:
            Tuple of (total_loss, loss_dict with components)
        """
        # Charbonnier pixel loss
        loss_pix = self.charbonnier(pred, target)
        
        # Gradient profile loss
        loss_grad = self.gradient(pred, target) if self.lambda_grad > 0 else torch.tensor(0.0, device=pred.device)
        
        # Perceptual loss - skip if lambda is 0 to avoid dtype issues
        loss_percep = self.perceptual(pred, target) if self.lambda_percep > 0 else torch.tensor(0.0, device=pred.device)
        
        # Semantic loss
        loss_sem = self.semantic(pred, target_text) if self.lambda_sem > 0 else torch.tensor(0.0, device=pred.device)
        
        # Total
        total = (
            self.lambda_pix * loss_pix +
            self.lambda_grad * loss_grad +
            self.lambda_percep * loss_percep +
            self.lambda_sem * loss_sem
        )
        
        loss_dict = {
            'loss_total': total.item(),
            'loss_pix': loss_pix.item(),
            'loss_grad': loss_grad.item() if isinstance(loss_grad, torch.Tensor) else 0.0,
            'loss_percep': loss_percep.item() if isinstance(loss_percep, torch.Tensor) else 0.0,
            'loss_sem': loss_sem.item() if isinstance(loss_sem, torch.Tensor) else 0.0,
        }
        
        return total, loss_dict


class PerceptualLoss(nn.Module):
    """
    Simplified perceptual loss using feature differences.
    
    Full version would use pretrained VGG.
    """
    
    def __init__(self):
        super().__init__()
        
        # Simple feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute perceptual loss."""
        pred_feat = self.features(pred)
        with torch.no_grad():
            target_feat = self.features(target)
        
        return F.l1_loss(pred_feat, target_feat)
