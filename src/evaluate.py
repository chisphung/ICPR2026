"""
Evaluation Module for MF-LPR².

Implements metrics for evaluating restoration and recognition quality:
- SSIM: Structural Similarity Index
- PSNR: Peak Signal-to-Noise Ratio
- Character Accuracy: Per-character recognition accuracy
- Plate Accuracy: Full plate recognition accuracy
"""
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from tqdm import tqdm

# Try to import skimage for SSIM/PSNR
try:
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import peak_signal_noise_ratio as psnr
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False


def compute_ssim(
    restored: np.ndarray,
    ground_truth: np.ndarray,
    channel_axis: int = -1,
) -> float:
    """
    Compute SSIM between restored and ground truth images.
    
    Args:
        restored: Restored image (H, W, C) or (H, W)
        ground_truth: Ground truth image (H, W, C) or (H, W)
        channel_axis: Axis for color channels
        
    Returns:
        SSIM score (higher is better, max=1.0)
    """
    if not SKIMAGE_AVAILABLE:
        return _ssim_fallback(restored, ground_truth)
    
    # Ensure same shape
    if restored.shape != ground_truth.shape:
        # Resize restored to match GT
        from PIL import Image
        restored = np.array(Image.fromarray(restored).resize(
            (ground_truth.shape[1], ground_truth.shape[0])
        ))
    
    # Handle channel axis for grayscale
    if restored.ndim == 2:
        return ssim(restored, ground_truth, data_range=255)
    
    return ssim(restored, ground_truth, data_range=255, channel_axis=channel_axis)


def compute_psnr(
    restored: np.ndarray,
    ground_truth: np.ndarray,
) -> float:
    """
    Compute PSNR between restored and ground truth images.
    
    Args:
        restored: Restored image (H, W, C) or (H, W)
        ground_truth: Ground truth image (H, W, C) or (H, W)
        
    Returns:
        PSNR score in dB (higher is better)
    """
    if not SKIMAGE_AVAILABLE:
        return _psnr_fallback(restored, ground_truth)
    
    # Ensure same shape
    if restored.shape != ground_truth.shape:
        from PIL import Image
        restored = np.array(Image.fromarray(restored).resize(
            (ground_truth.shape[1], ground_truth.shape[0])
        ))
    
    return psnr(ground_truth, restored, data_range=255)


def _ssim_fallback(img1: np.ndarray, img2: np.ndarray) -> float:
    """Simple SSIM fallback when skimage not available."""
    # Convert to float
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    
    mu1 = img1.mean()
    mu2 = img2.mean()
    sigma1_sq = ((img1 - mu1) ** 2).mean()
    sigma2_sq = ((img2 - mu2) ** 2).mean()
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()
    
    ssim_val = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
               ((mu1 ** 2 + mu2 ** 2 + c1) * (sigma1_sq + sigma2_sq + c2))
    
    return float(ssim_val)


def _psnr_fallback(img1: np.ndarray, img2: np.ndarray) -> float:
    """Simple PSNR fallback when skimage not available."""
    mse = ((img1.astype(np.float64) - img2.astype(np.float64)) ** 2).mean()
    if mse == 0:
        return float('inf')
    return 10 * np.log10(255 ** 2 / mse)


def compute_character_accuracy(prediction: str, ground_truth: str) -> Tuple[float, int, int]:
    """
    Compute per-character accuracy.
    
    Args:
        prediction: Predicted text
        ground_truth: Ground truth text
        
    Returns:
        Tuple of (accuracy, correct_chars, total_chars)
    """
    # Normalize: uppercase and remove spaces
    pred = prediction.upper().replace(" ", "").replace("-", "")
    gt = ground_truth.upper().replace(" ", "").replace("-", "")
    
    if len(gt) == 0:
        return 1.0 if len(pred) == 0 else 0.0, 0, 0
    
    # Count matching characters at same positions
    correct = 0
    for i, char in enumerate(gt):
        if i < len(pred) and pred[i] == char:
            correct += 1
    
    return correct / len(gt), correct, len(gt)


def compute_plate_accuracy(prediction: str, ground_truth: str) -> bool:
    """
    Check if plate is correctly recognized.
    
    Args:
        prediction: Predicted text
        ground_truth: Ground truth text
        
    Returns:
        True if prediction matches ground truth
    """
    pred = prediction.upper().replace(" ", "").replace("-", "")
    gt = ground_truth.upper().replace(" ", "").replace("-", "")
    return pred == gt


def evaluate_track(
    restored_image: np.ndarray,
    ground_truth_image: np.ndarray,
    predicted_text: str,
    ground_truth_text: str,
) -> Dict:
    """
    Evaluate a single track.
    
    Args:
        restored_image: Restored image from pipeline
        ground_truth_image: High-resolution ground truth
        predicted_text: Predicted plate text
        ground_truth_text: Ground truth plate text
        
    Returns:
        Dict with all metrics
    """
    # Image quality metrics
    ssim_score = compute_ssim(restored_image, ground_truth_image)
    psnr_score = compute_psnr(restored_image, ground_truth_image)
    
    # Text recognition metrics
    char_acc, correct, total = compute_character_accuracy(predicted_text, ground_truth_text)
    plate_correct = compute_plate_accuracy(predicted_text, ground_truth_text)
    
    return {
        'ssim': ssim_score,
        'psnr': psnr_score,
        'char_accuracy': char_acc,
        'correct_chars': correct,
        'total_chars': total,
        'plate_correct': plate_correct,
        'predicted_text': predicted_text,
        'ground_truth_text': ground_truth_text,
    }


def evaluate_dataset(
    results: List[Dict],
) -> Dict:
    """
    Aggregate evaluation results over entire dataset.
    
    Args:
        results: List of per-track evaluation results
        
    Returns:
        Aggregated metrics
    """
    if not results:
        return {}
    
    ssim_scores = [r['ssim'] for r in results if 'ssim' in r]
    psnr_scores = [r['psnr'] for r in results if 'psnr' in r and r['psnr'] != float('inf')]
    char_accs = [r['char_accuracy'] for r in results if 'char_accuracy' in r]
    plate_correct = [r['plate_correct'] for r in results if 'plate_correct' in r]
    
    total_correct_chars = sum(r.get('correct_chars', 0) for r in results)
    total_chars = sum(r.get('total_chars', 0) for r in results)
    
    return {
        'ssim_mean': np.mean(ssim_scores) if ssim_scores else 0,
        'ssim_std': np.std(ssim_scores) if ssim_scores else 0,
        'psnr_mean': np.mean(psnr_scores) if psnr_scores else 0,
        'psnr_std': np.std(psnr_scores) if psnr_scores else 0,
        'char_accuracy_mean': np.mean(char_accs) if char_accs else 0,
        'char_accuracy_overall': total_correct_chars / total_chars if total_chars > 0 else 0,
        'plate_accuracy': np.mean(plate_correct) if plate_correct else 0,
        'num_samples': len(results),
    }


def print_evaluation_report(metrics: Dict):
    """Print formatted evaluation report."""
    print("\n" + "=" * 50)
    print("MF-LPR² Evaluation Report")
    print("=" * 50)
    print(f"Number of samples: {metrics.get('num_samples', 0)}")
    print("-" * 50)
    print("Image Quality Metrics:")
    print(f"  SSIM:  {metrics.get('ssim_mean', 0):.4f} ± {metrics.get('ssim_std', 0):.4f}")
    print(f"  PSNR:  {metrics.get('psnr_mean', 0):.2f} ± {metrics.get('psnr_std', 0):.2f} dB")
    print("-" * 50)
    print("Recognition Metrics:")
    print(f"  Character Accuracy: {metrics.get('char_accuracy_overall', 0)*100:.2f}%")
    print(f"  Plate Accuracy:     {metrics.get('plate_accuracy', 0)*100:.2f}%")
    print("=" * 50)
