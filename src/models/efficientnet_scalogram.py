"""
EfficientNet + Scalogram Model - Paper 2 (2D Vision Classification)

AttentionEfficientNet with:
- EfficientNet-B0 backbone (pretrained on ImageNet)
- CBAM attention module
- Custom head for ECG classification
- Built-in CWT scalogram generation

Input: 1D ECG signal -> CWT Scalogram -> EfficientNet -> Classification

Reference:
- Tan & Le, "EfficientNet: Rethinking Model Scaling for CNNs"
- Woo et al., "CBAM: Convolutional Block Attention Module"

Usage:
    from src.models import AttentionEfficientNet
    
    model = AttentionEfficientNet(num_classes=5)
    # Input: raw 1D signals
    out = model(x, precomputed_scalogram=False)
    # Input: precomputed scalograms
    out = model(scalogram_images, precomputed_scalogram=True)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pywt
import cv2
from typing import Optional, Tuple

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False


# ============================================================================
# CWT SCALOGRAM LAYER
# ============================================================================

class CWTScalogramLayer(nn.Module):
    """
    Converts 1D signals to 2D scalograms using Continuous Wavelet Transform.
    
    Can run on GPU via torch operations for batch processing.
    
    Args:
        scales: Wavelet scales (default: 1-64)
        wavelet: Wavelet type (supports 'morl', 'mexh', 'cmor')
        img_size: Output image size (square)
    """
    
    def __init__(self, scales: int = 64, wavelet: str = 'morl', img_size: int = 224):
        super().__init__()
        self.scales = np.arange(1, scales + 1)
        self.wavelet = wavelet
        self.img_size = img_size
        
        # ImageNet normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert batch of 1D signals to batch of 2D scalogram images.
        
        Args:
            x: (batch, length) or (batch, 1, length)
            
        Returns:
            (batch, 3, img_size, img_size) - RGB scalogram images
        """
        if x.dim() == 3:
            x = x.squeeze(1)
        
        batch_size = x.shape[0]
        device = x.device
        
        # Process on CPU (pywt requires numpy)
        x_np = x.cpu().numpy()
        
        scalograms = []
        for i in range(batch_size):
            # Compute CWT
            coeffs, _ = pywt.cwt(x_np[i], self.scales, self.wavelet)
            scalo = np.abs(coeffs)
            
            # Resize
            scalo_resized = cv2.resize(scalo, (self.img_size, self.img_size),
                                       interpolation=cv2.INTER_CUBIC)
            
            # Normalize to 0-1
            scalo_norm = (scalo_resized - scalo_resized.min()) / \
                        (scalo_resized.max() - scalo_resized.min() + 1e-8)
            
            scalograms.append(scalo_norm)
        
        # Stack and convert to RGB
        scalograms = np.stack(scalograms)  # (B, H, W)
        scalograms = np.stack([scalograms] * 3, axis=1)  # (B, 3, H, W)
        
        # Convert to tensor
        img_tensor = torch.from_numpy(scalograms.astype(np.float32)).to(device)
        
        # Apply ImageNet normalization
        img_tensor = (img_tensor - self.mean) / self.std
        
        return img_tensor


# ============================================================================
# ATTENTION MODULES
# ============================================================================

class ChannelAttention(nn.Module):
    """Channel attention module from CBAM."""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return torch.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """Spatial attention module from CBAM."""
    
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
    
    def forward(self, x):
        avg_out = x.mean(dim=1, keepdim=True)
        max_out = x.max(dim=1, keepdim=True)[0]
        concat = torch.cat([avg_out, max_out], dim=1)
        return torch.sigmoid(self.conv(concat))


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.
    
    Applies channel attention followed by spatial attention.
    """
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.channel_attn = ChannelAttention(channels, reduction)
        self.spatial_attn = SpatialAttention()
    
    def forward(self, x):
        x = x * self.channel_attn(x)
        x = x * self.spatial_attn(x)
        return x


# ============================================================================
# MAIN MODEL
# ============================================================================

class AttentionEfficientNet(nn.Module):
    """
    EfficientNet with CBAM attention for ECG scalogram classification.
    
    Features:
    - EfficientNet-B0 backbone (can swap for B1-B7)
    - CBAM attention after feature extraction
    - Multi-scale feature fusion
    - Built-in CWT scalogram generation
    
    Args:
        num_classes: Number of output classes
        backbone: EfficientNet variant ('b0', 'b1', 'b2', etc.)
        pretrained: Use ImageNet pretrained weights
        dropout: Dropout rate
        cbam_reduction: CBAM channel reduction factor
        use_compile: Use torch.compile()
    """
    
    def __init__(self,
                 num_classes: int = 5,
                 backbone: str = 'b0',
                 pretrained: bool = True,
                 dropout: float = 0.3,
                 cbam_reduction: int = 16,
                 img_size: int = 224,
                 use_compile: bool = False):
        super().__init__()
        
        if not HAS_TIMM:
            raise ImportError("timm required. Install: pip install timm")
        
        self.num_classes = num_classes
        self.img_size = img_size
        
        # Scalogram layer
        self.cwt_layer = CWTScalogramLayer(scales=64, img_size=img_size)
        
        # EfficientNet backbone
        model_name = f'efficientnet_{backbone}'
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(2, 3, 4)  # Extract 3 scales
        )
        
        # Get feature channels for each scale
        with torch.no_grad():
            dummy = torch.zeros(1, 3, img_size, img_size)
            feat_channels = [f.shape[1] for f in self.backbone(dummy)]
        
        # CBAM for each scale
        self.cbam_modules = nn.ModuleList([
            CBAM(ch, cbam_reduction) for ch in feat_channels
        ])
        
        # Feature fusion
        fused_channels = sum(feat_channels)
        self.fusion = nn.Sequential(
            nn.Conv2d(fused_channels, 512, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            CBAM(512, cbam_reduction)
        )
        
        # Classification head
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_classes)
        )
        
        # Feature dimension for meta-learning
        self.feature_dim = 512 * 2
        
        # Initialize custom layers
        self._init_weights()
        
        if use_compile and hasattr(torch, 'compile'):
            self._forward_impl = torch.compile(self._forward_impl)
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) and m not in self.backbone.modules():
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d) and m not in self.backbone.modules():
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass on scalogram images."""
        # Extract multi-scale features
        features = self.backbone(x)
        
        # Apply CBAM attention to each scale
        attented = [cbam(f) for cbam, f in zip(self.cbam_modules, features)]
        
        # Upsample to same size and concatenate
        target_size = attented[0].shape[2:]
        fused = torch.cat([
            F.interpolate(f, size=target_size, mode='bilinear', align_corners=False)
            for f in attented
        ], dim=1)
        
        # Apply fusion
        fused = self.fusion(fused)
        
        # Global pooling
        avg_pool = self.gap(fused)
        max_pool = self.gmp(fused)
        pooled = torch.cat([avg_pool, max_pool], dim=1)
        
        # Classify
        return self.classifier(pooled)
    
    def forward(self, x: torch.Tensor, precomputed_scalogram: bool = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor. Either:
               - Raw signals: (batch, length) or (batch, 1, length)
               - Precomputed scalograms: (batch, 3, H, W)
            precomputed_scalogram: If True, skip CWT conversion. 
                                   If None, auto-detect from input shape.
            
        Returns:
            Logits of shape (batch, num_classes)
        """
        # Auto-detect: if input has 4 dims and channel=3, it's a pre-computed scalogram
        if precomputed_scalogram is None:
            precomputed_scalogram = (x.dim() == 4 and x.shape[1] == 3)
        
        if not precomputed_scalogram:
            x = self.cwt_layer(x)
        
        return self._forward_impl(x)
    
    def get_features(self, x: torch.Tensor, precomputed_scalogram: bool = None) -> torch.Tensor:
        """Extract features before classifier."""
        # Auto-detect
        if precomputed_scalogram is None:
            precomputed_scalogram = (x.dim() == 4 and x.shape[1] == 3)
            
        if not precomputed_scalogram:
            x = self.cwt_layer(x)
        
        features = self.backbone(x)
        attented = [cbam(f) for cbam, f in zip(self.cbam_modules, features)]
        
        target_size = attented[0].shape[2:]
        fused = torch.cat([
            F.interpolate(f, size=target_size, mode='bilinear', align_corners=False)
            for f in attented
        ], dim=1)
        
        fused = self.fusion(fused)
        
        avg_pool = self.gap(fused).flatten(1)
        max_pool = self.gmp(fused).flatten(1)
        
        return torch.cat([avg_pool, max_pool], dim=1)


# ============================================================================
# MODEL FACTORY
# ============================================================================

def create_efficientnet_scalogram(num_classes: int = 5,
                                  variant: str = 'b0',
                                  pretrained: bool = True,
                                  compile: bool = False) -> AttentionEfficientNet:
    """
    Factory function for EfficientNet variants.
    
    Args:
        num_classes: Number of classes
        variant: EfficientNet variant ('b0' to 'b7')
        pretrained: Use ImageNet weights
        compile: Use torch.compile()
    """
    return AttentionEfficientNet(
        num_classes=num_classes,
        backbone=variant,
        pretrained=pretrained,
        use_compile=compile
    )


if __name__ == "__main__":
    # Quick test
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = create_efficientnet_scalogram(num_classes=5, variant='b0')
    model = model.to(device)
    
    # Test with raw signals
    x_raw = torch.randn(4, 360).to(device)
    out = model(x_raw, precomputed_scalogram=False)
    
    print(f"Model: AttentionEfficientNet")
    print(f"Input (raw signal): {x_raw.shape}")
    print(f"Output: {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test feature extraction
    features = model.get_features(x_raw)
    print(f"Features shape: {features.shape}")
