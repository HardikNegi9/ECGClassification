"""
Dual Ensemble Model - Signal + Vision Fusion

Fuses two Deep Learning Giants:
1. Signal Expert (Context-Aware InceptionTime) - 1D temporal processing
2. Vision Expert (EfficientNet-B0 + CBAM) - 2D scalogram processing

Fusion strategies:
- Weighted average (default: 60% signal, 40% vision)
- Learnable fusion (MLP on concatenated logits/features)
- Attention-based fusion

Usage:
    from src.models import DualEnsemble
    
    # Option 1: Use with pre-trained models
    ensemble = DualEnsemble.from_pretrained(
        signal_path='models/signal_expert.pt',
        vision_path='models/vision_expert.pt'
    )
    probs = ensemble.predict(signals)
    
    # Option 2: End-to-end training
    ensemble = DualEnsemble(num_classes=5, fusion='learnable')
    out = ensemble(signals)  # handles scalogram conversion internally
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision import transforms
import pywt
import cv2
from typing import Dict, Tuple, Optional, List
from tqdm import tqdm

# Import base models
from .inception_time import ContextAwareInceptionTime, create_inception_time
from .efficientnet_scalogram import AttentionEfficientNet, create_efficientnet_scalogram


# ============================================================================
# SCALOGRAM CONVERSION
# ============================================================================

class ScalogramConverter:
    """Converts 1D signals to 2D scalograms for Vision Expert."""
    
    def __init__(self, img_size: int = 224, scales: np.ndarray = None, 
                 wavelet: str = 'morl'):
        self.img_size = img_size
        self.scales = scales if scales is not None else np.arange(1, 65)
        self.wavelet = wavelet
        
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def __call__(self, signal: np.ndarray) -> torch.Tensor:
        """Convert single signal to scalogram tensor."""
        coeffs, _ = pywt.cwt(signal, self.scales, self.wavelet)
        scalogram = np.abs(coeffs)
        
        resized = cv2.resize(scalogram, (self.img_size, self.img_size), 
                            interpolation=cv2.INTER_CUBIC)
        resized = (resized - resized.min()) / (resized.max() - resized.min() + 1e-8)
        
        rgb_img = np.stack((resized,) * 3, axis=-1).astype(np.float32)
        return self.normalize(rgb_img)
    
    def batch_convert(self, signals: np.ndarray, device: torch.device = None) -> torch.Tensor:
        """Convert batch of signals to scalogram tensors."""
        scalograms = []
        for sig in signals:
            scalograms.append(self(sig))
        batch = torch.stack(scalograms)
        if device:
            batch = batch.to(device)
        return batch


# ============================================================================
# FUSION MODULES
# ============================================================================

class WeightedFusion(nn.Module):
    """Simple weighted average fusion."""
    
    def __init__(self, signal_weight: float = 0.60, vision_weight: float = 0.40):
        super().__init__()
        self.register_buffer('signal_weight', torch.tensor(signal_weight))
        self.register_buffer('vision_weight', torch.tensor(vision_weight))
    
    def forward(self, signal_logits: torch.Tensor, vision_logits: torch.Tensor) -> torch.Tensor:
        signal_probs = F.softmax(signal_logits, dim=1)
        vision_probs = F.softmax(vision_logits, dim=1)
        fused = signal_probs * self.signal_weight + vision_probs * self.vision_weight
        return fused  # Return probabilities


class LearnableFusion(nn.Module):
    """Learnable fusion with MLP on concatenated logits."""
    
    def __init__(self, num_classes: int = 5, hidden_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        input_dim = num_classes * 2  # Concatenated logits from both experts
        
        self.fusion = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, signal_logits: torch.Tensor, vision_logits: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([signal_logits, vision_logits], dim=1)
        return self.fusion(combined)


class AttentionFusion(nn.Module):
    """Attention-based fusion learning to weight each expert per sample."""
    
    def __init__(self, num_classes: int = 5, hidden_dim: int = 32):
        super().__init__()
        input_dim = num_classes * 2
        
        # Attention network to compute dynamic weights
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2),  # 2 experts
            nn.Softmax(dim=1)
        )
    
    def forward(self, signal_logits: torch.Tensor, vision_logits: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([signal_logits, vision_logits], dim=1)
        
        # Get attention weights for this sample
        weights = self.attention(combined)  # (B, 2)
        
        # Apply weights
        signal_probs = F.softmax(signal_logits, dim=1)
        vision_probs = F.softmax(vision_logits, dim=1)
        
        fused = weights[:, 0:1] * signal_probs + weights[:, 1:2] * vision_probs
        return fused


class FeatureFusion(nn.Module):
    """Fusion on feature level (before classification heads)."""
    
    def __init__(self, signal_feat_dim: int, vision_feat_dim: int, 
                 num_classes: int = 5, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        
        combined_dim = signal_feat_dim + vision_feat_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, signal_features: torch.Tensor, vision_features: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([signal_features, vision_features], dim=1)
        return self.classifier(combined)


# ============================================================================
# DUAL ENSEMBLE MODEL
# ============================================================================

class DualEnsemble(nn.Module):
    """
    Dual Expert Ensemble combining Signal and Vision models.
    
    Args:
        num_classes: Number of output classes
        fusion: Fusion strategy ('weighted', 'learnable', 'attention', 'feature')
        signal_weight: Weight for signal expert (only for 'weighted' fusion)
        vision_weight: Weight for vision expert (only for 'weighted' fusion)
        freeze_experts: Freeze expert weights during training
        signal_variant: InceptionTime variant ('tiny', 'base', 'large')
        vision_variant: EfficientNet variant ('b0', 'b1', etc.)
    """
    
    FUSION_TYPES = ['weighted', 'learnable', 'attention', 'feature']
    
    def __init__(self,
                 num_classes: int = 5,
                 fusion: str = 'weighted',
                 signal_weight: float = 0.60,
                 vision_weight: float = 0.40,
                 freeze_experts: bool = False,
                 signal_variant: str = 'base',
                 vision_variant: str = 'b0',
                 img_size: int = 224):
        super().__init__()
        
        if fusion not in self.FUSION_TYPES:
            raise ValueError(f"Unknown fusion: {fusion}. Choose from {self.FUSION_TYPES}")
        
        self.num_classes = num_classes
        self.fusion_type = fusion
        self.freeze_experts = freeze_experts
        self.img_size = img_size
        
        # Expert models
        self.signal_expert = create_inception_time(num_classes, variant=signal_variant)
        self.vision_expert = create_efficientnet_scalogram(num_classes, variant=vision_variant, 
                                                           pretrained=True)
        
        # Scalogram converter
        self.scalogram_converter = ScalogramConverter(img_size=img_size)
        
        # Fusion module
        if fusion == 'weighted':
            self.fusion = WeightedFusion(signal_weight, vision_weight)
        elif fusion == 'learnable':
            self.fusion = LearnableFusion(num_classes)
        elif fusion == 'attention':
            self.fusion = AttentionFusion(num_classes)
        elif fusion == 'feature':
            # Get feature dimensions from experts
            signal_feat_dim = self.signal_expert.feature_dim if hasattr(self.signal_expert, 'feature_dim') else 256
            vision_feat_dim = self.vision_expert.feature_dim if hasattr(self.vision_expert, 'feature_dim') else 1024
            self.fusion = FeatureFusion(signal_feat_dim, vision_feat_dim, num_classes)
        
        if freeze_experts:
            self._freeze_experts()
    
    def _freeze_experts(self):
        """Freeze expert model parameters."""
        for param in self.signal_expert.parameters():
            param.requires_grad = False
        for param in self.vision_expert.parameters():
            param.requires_grad = False
    
    def _unfreeze_experts(self):
        """Unfreeze expert model parameters."""
        for param in self.signal_expert.parameters():
            param.requires_grad = True
        for param in self.vision_expert.parameters():
            param.requires_grad = True
    
    def load_expert_weights(self, signal_path: str = None, vision_path: str = None,
                           device: torch.device = None):
        """Load pre-trained weights for experts."""
        device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if signal_path and os.path.exists(signal_path):
            state_dict = torch.load(signal_path, map_location=device)
            state_dict = self._clean_state_dict(state_dict)
            self.signal_expert.load_state_dict(state_dict)
            print(f"Loaded Signal Expert from: {signal_path}")
        
        if vision_path and os.path.exists(vision_path):
            state_dict = torch.load(vision_path, map_location=device)
            state_dict = self._clean_state_dict(state_dict)
            self.vision_expert.load_state_dict(state_dict)
            print(f"Loaded Vision Expert from: {vision_path}")
    
    @staticmethod
    def _clean_state_dict(state_dict: dict) -> dict:
        """Remove '_orig_mod.' prefix from torch.compile state dicts."""
        cleaned = {}
        for k, v in state_dict.items():
            if k.startswith("_orig_mod."):
                k = k[10:]
            cleaned[k] = v
        return cleaned
    
    def forward(self, x: torch.Tensor, scalograms: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: 1D signals (batch, length) or (batch, 1, length)
            scalograms: Pre-computed scalograms (batch, 3, H, W). If None, computed on-the-fly.
            
        Returns:
            Logits or probabilities depending on fusion type
        """
        # Signal Expert
        signal_out = self.signal_expert(x)
        
        # Vision Expert
        if scalograms is not None:
            vision_out = self.vision_expert(scalograms, precomputed_scalogram=True)
        else:
            vision_out = self.vision_expert(x, precomputed_scalogram=False)
        
        # Fusion
        if self.fusion_type == 'feature':
            # For feature fusion, get features instead of logits
            signal_feat = self.signal_expert.get_features(x)
            vision_feat = self.vision_expert.get_features(x)
            return self.fusion(signal_feat, vision_feat)
        else:
            return self.fusion(signal_out, vision_out)
    
    @classmethod
    def from_pretrained(cls, signal_path: str, vision_path: str,
                       num_classes: int = 5, fusion: str = 'weighted',
                       freeze_experts: bool = True, **kwargs) -> 'DualEnsemble':
        """
        Create ensemble from pre-trained expert models.
        
        Args:
            signal_path: Path to Signal Expert weights
            vision_path: Path to Vision Expert weights
            num_classes: Number of classes
            fusion: Fusion strategy
            freeze_experts: Freeze expert weights
            
        Returns:
            DualEnsemble with loaded weights
        """
        model = cls(
            num_classes=num_classes,
            fusion=fusion,
            freeze_experts=freeze_experts,
            **kwargs
        )
        model.load_expert_weights(signal_path, vision_path)
        return model
    
    @torch.no_grad()
    def predict(self, signals: np.ndarray, batch_size: int = 256,
                device: torch.device = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict on numpy array of signals.
        
        Args:
            signals: (N, length) array of ECG signals
            batch_size: Batch size for inference
            device: torch.device
            
        Returns:
            predictions: (N,) predicted class labels
            probabilities: (N, num_classes) prediction probabilities
        """
        device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        self.eval()
        
        all_probs = []
        
        n_samples = len(signals)
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_signals = signals[start:end]
            
            # Convert to tensor
            x = torch.FloatTensor(batch_signals).to(device)
            
            # Forward
            out = self(x)
            
            # Get probabilities
            if self.fusion_type in ['learnable', 'feature']:
                probs = F.softmax(out, dim=1)
            else:
                probs = out  # Already probabilities for weighted/attention
            
            all_probs.append(probs.cpu().numpy())
        
        probabilities = np.vstack(all_probs)
        predictions = probabilities.argmax(axis=1)
        
        return predictions, probabilities


# ============================================================================
# ENSEMBLE INFERENCE (Standalone Function)
# ============================================================================

@torch.no_grad()
def ensemble_inference(signal_model: nn.Module,
                       vision_model: nn.Module,
                       signals: np.ndarray,
                       y_true: np.ndarray = None,
                       signal_weight: float = 0.60,
                       vision_weight: float = 0.40,
                       batch_size_signal: int = 512,
                       batch_size_vision: int = 128,
                       device: torch.device = None,
                       verbose: bool = True) -> Dict:
    """
    Run inference with pre-loaded signal and vision models.
    
    This is the standalone function for quick ensemble evaluation.
    
    Args:
        signal_model: Loaded Signal Expert model
        vision_model: Loaded Vision Expert model
        signals: (N, length) ECG signals
        y_true: Ground truth labels (optional, for metrics)
        signal_weight: Weight for signal expert
        vision_weight: Weight for vision expert
        batch_size_signal: Batch size for signal model
        batch_size_vision: Batch size for vision model
        device: torch.device
        verbose: Print progress
        
    Returns:
        Dict with predictions, probabilities, and optionally metrics
    """
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    signal_model = signal_model.to(device).eval()
    vision_model = vision_model.to(device).eval()
    
    # Signal Expert Inference
    if verbose:
        print("Running Signal Expert...")
    
    ds_sig = TensorDataset(torch.FloatTensor(signals))
    dl_sig = DataLoader(ds_sig, batch_size=batch_size_signal, shuffle=False)
    
    probs_signal = []
    iterator = tqdm(dl_sig, desc="Signal") if verbose else dl_sig
    for (xb,) in iterator:
        out = signal_model(xb.to(device))
        probs_signal.append(F.softmax(out, dim=1).cpu().numpy())
    probs_signal = np.vstack(probs_signal)
    
    # Vision Expert Inference
    if verbose:
        print("Running Vision Expert...")
    
    class SignalScalogramDataset(Dataset):
        def __init__(self, X, img_size=224):
            self.X = X
            self.converter = ScalogramConverter(img_size)
        def __len__(self):
            return len(self.X)
        def __getitem__(self, idx):
            return self.converter(self.X[idx])
    
    ds_vis = SignalScalogramDataset(signals)
    dl_vis = DataLoader(ds_vis, batch_size=batch_size_vision, shuffle=False, num_workers=0)
    
    probs_vision = []
    iterator = tqdm(dl_vis, desc="Vision") if verbose else dl_vis
    for xb in iterator:
        out = vision_model(xb.to(device))
        probs_vision.append(F.softmax(out, dim=1).cpu().numpy())
    probs_vision = np.vstack(probs_vision)
    
    # Fusion
    final_probs = probs_signal * signal_weight + probs_vision * vision_weight
    final_preds = final_probs.argmax(axis=1)
    
    result = {
        'predictions': final_preds,
        'probabilities': final_probs,
        'probs_signal': probs_signal,
        'probs_vision': probs_vision,
        'signal_weight': signal_weight,
        'vision_weight': vision_weight,
    }
    
    # Compute metrics if labels provided
    if y_true is not None:
        acc = accuracy_score(y_true, final_preds)
        result['accuracy'] = acc
        result['classification_report'] = classification_report(
            y_true, final_preds, target_names=['N', 'S', 'V', 'F', 'Q'], digits=4
        )
        result['confusion_matrix'] = confusion_matrix(y_true, final_preds)
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"DUAL ENSEMBLE ACCURACY: {acc*100:.4f}%")
            print(f"{'='*60}")
            print(f"\n{result['classification_report']}")
    
    return result


# ============================================================================
# MODEL FACTORY
# ============================================================================

def create_dual_ensemble(num_classes: int = 5,
                        fusion: str = 'weighted',
                        signal_weight: float = 0.60,
                        vision_weight: float = 0.40,
                        signal_path: str = None,
                        vision_path: str = None,
                        freeze_experts: bool = False) -> DualEnsemble:
    """
    Factory function for Dual Ensemble.
    
    Args:
        num_classes: Number of classes
        fusion: Fusion strategy
        signal_weight: Weight for signal expert
        vision_weight: Weight for vision expert
        signal_path: Path to pretrained signal model
        vision_path: Path to pretrained vision model
        freeze_experts: Freeze expert weights
        
    Returns:
        DualEnsemble model
    """
    if signal_path and vision_path:
        return DualEnsemble.from_pretrained(
            signal_path=signal_path,
            vision_path=vision_path,
            num_classes=num_classes,
            fusion=fusion,
            freeze_experts=freeze_experts
        )
    else:
        return DualEnsemble(
            num_classes=num_classes,
            fusion=fusion,
            signal_weight=signal_weight,
            vision_weight=vision_weight,
            freeze_experts=freeze_experts
        )


if __name__ == "__main__":
    # Quick test
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test weighted fusion ensemble
    model = DualEnsemble(num_classes=5, fusion='weighted')
    model = model.to(device)
    
    # Test with synthetic data
    x = torch.randn(4, 360).to(device)
    out = model(x)
    
    print(f"Model: DualEnsemble (weighted fusion)")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test learnable fusion
    model_learn = DualEnsemble(num_classes=5, fusion='learnable')
    model_learn = model_learn.to(device)
    out_learn = model_learn(x)
    print(f"\nLearnable fusion output: {out_learn.shape}")
