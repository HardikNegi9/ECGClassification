"""
NSHT Dual-Evolution Model - Paper 3 (Hybrid Multi-Scale Classification)

Neural Spectral-Hybrid Transformer with Dual Evolution:
- Learnable Adaptive Morlet Wavelets (frequency domain learning)
- Dual-stream 1D + 2D processing
- Coordinate Attention for spatial-spectral fusion
- Evolutionary fusion of temporal and spectral features

This model combines the best of 1D signal processing (InceptionTime)
with 2D vision models (EfficientNet) in a unified architecture.

Usage:
    from src.models import NSHT_Dual_Evo
    
    model = NSHT_Dual_Evo(num_classes=5)
    out = model(x)  # x: (batch, 360)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Tuple


# ============================================================================
# LEARNABLE WAVELET TRANSFORM
# ============================================================================

class AdaptiveMorletWavelet(nn.Module):
    """
    Learnable Morlet Wavelet for adaptive time-frequency decomposition.
    
    Parameters (learnable):
    - sigma: Controls time-frequency trade-off
    - omega0: Central frequency
    - scale factors: Multi-scale decomposition
    """
    
    def __init__(self, n_scales: int = 32, signal_length: int = 360):
        super().__init__()
        
        self.n_scales = n_scales
        self.signal_length = signal_length
        
        # Learnable parameters
        self.sigma = nn.Parameter(torch.ones(n_scales) * 1.0)
        self.omega0 = nn.Parameter(torch.linspace(0.5, 6.0, n_scales))
        
        # Learnable scales (log-space for stability)
        self.log_scales = nn.Parameter(torch.linspace(0, 3, n_scales))
        
        # Time axis
        t = torch.linspace(-4, 4, signal_length)
        self.register_buffer('t', t)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply learnable wavelet transform.
        
        Args:
            x: (batch, length) or (batch, 1, length)
            
        Returns:
            Scalogram: (batch, n_scales, length)
        """
        if x.dim() == 3:
            x = x.squeeze(1)
        
        batch_size = x.shape[0]
        
        # Get scales
        scales = torch.exp(self.log_scales)  # Ensure positive
        
        # Compute Morlet wavelets
        wavelets = []
        for i in range(self.n_scales):
            # Morlet: exp(i*omega0*t) * exp(-t^2/(2*sigma^2))
            s = scales[i]
            t_scaled = self.t / s
            
            # Complex Morlet wavelet
            gaussian = torch.exp(-t_scaled ** 2 / (2 * self.sigma[i] ** 2 + 1e-6))
            oscillation_real = torch.cos(self.omega0[i] * t_scaled)
            oscillation_imag = torch.sin(self.omega0[i] * t_scaled)
            
            wavelet_real = gaussian * oscillation_real / torch.sqrt(s + 1e-6)
            wavelet_imag = gaussian * oscillation_imag / torch.sqrt(s + 1e-6)
            
            wavelets.append(torch.stack([wavelet_real, wavelet_imag], dim=0))
        
        wavelets = torch.stack(wavelets)  # (n_scales, 2, length)
        
        # Convolve with signal (batch-wise)
        # Use circular padding for scale invariance
        x_padded = F.pad(x, (self.signal_length // 2, self.signal_length // 2), mode='circular')
        
        # Reshape for conv1d: (batch, 1, padded_length)
        x_padded = x_padded.unsqueeze(1)
        
        # Flip wavelets for convolution
        wavelets_real = wavelets[:, 0, :].flip(-1).unsqueeze(1)  # (n_scales, 1, length)
        wavelets_imag = wavelets[:, 1, :].flip(-1).unsqueeze(1)
        
        # Convolve
        cwt_real = F.conv1d(x_padded, wavelets_real)  # (batch, n_scales, out_length)
        cwt_imag = F.conv1d(x_padded, wavelets_imag)
        
        # Compute magnitude (abs of complex)
        cwt_mag = torch.sqrt(cwt_real ** 2 + cwt_imag ** 2 + 1e-8)
        
        # Trim to original length
        start = (cwt_mag.shape[2] - self.signal_length) // 2
        cwt_mag = cwt_mag[:, :, start:start + self.signal_length]
        
        return cwt_mag


# ============================================================================
# ATTENTION MODULES
# ============================================================================

class CoordinateAttention(nn.Module):
    """
    Coordinate Attention for capturing long-range dependencies.
    
    Decomposes channel attention into two 1D feature encoding
    along different spatial directions.
    """
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        
        reduced = max(8, channels // reduction)
        
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        
        self.conv1 = nn.Conv2d(channels, reduced, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(reduced)
        self.act = nn.Hardswish()
        
        self.conv_h = nn.Conv2d(reduced, channels, 1, bias=False)
        self.conv_w = nn.Conv2d(reduced, channels, 1, bias=False)
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Pool along H and W
        x_h = self.pool_h(x)  # (B, C, H, 1)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # (B, C, W, 1)
        
        # Concatenate
        y = torch.cat([x_h, x_w], dim=2)  # (B, C, H+W, 1)
        y = self.act(self.bn1(self.conv1(y)))
        
        # Split
        x_h, x_w = torch.split(y, [H, W], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        
        # Attention maps
        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w))
        
        return x * a_h * a_w


class TemporalAttention(nn.Module):
    """Multi-head self-attention for 1D signals."""
    
    def __init__(self, dim: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)
    
    def forward(self, x):
        # x: (B, L, C)
        B, L, C = x.shape
        
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, L, C)
        x = self.proj_drop(self.proj(x))
        
        return x


# ============================================================================
# STREAM MODULES
# ============================================================================

class TemporalStream(nn.Module):
    """1D temporal stream with inception-style processing."""
    
    def __init__(self, out_dim: int = 256):
        super().__init__()
        
        # Multi-scale 1D convolutions
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )
        
        # Inception-style block
        self.inception = nn.ModuleList([
            nn.Conv1d(64, 32, 9, padding=4),
            nn.Conv1d(64, 32, 19, padding=9),
            nn.Conv1d(64, 32, 39, padding=19),
        ])
        
        self.bn = nn.BatchNorm1d(96)
        
        # Attention
        self.attention = TemporalAttention(96, n_heads=4)
        
        # Project to output dim
        self.project = nn.Sequential(
            nn.Conv1d(96, out_dim, 1),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # x: (B, L) or (B, 1, L)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        x = self.conv1(x)
        
        # Inception
        branches = [conv(x) for conv in self.inception]
        x = torch.cat(branches, dim=1)
        x = F.relu(self.bn(x))
        
        # Attention (transpose for attention)
        x = x.transpose(1, 2)  # (B, L, C)
        x = x + self.attention(x)
        x = x.transpose(1, 2)  # (B, C, L)
        
        # Project
        x = self.project(x)
        
        return x  # (B, out_dim, L)


class SpectralStream(nn.Module):
    """2D spectral stream with learnable wavelet and coordinate attention."""
    
    def __init__(self, n_scales: int = 32, signal_length: int = 360, out_dim: int = 256):
        super().__init__()
        
        # Learnable wavelet
        self.wavelet = AdaptiveMorletWavelet(n_scales, signal_length)
        
        # 2D processing
        self.conv2d = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        # Coordinate attention
        self.coord_attn = CoordinateAttention(128)
        
        # More processing
        self.conv2d_2 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            CoordinateAttention(256)
        )
        
        # Reduce to 1D
        self.reduce = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, None)),  # (B, C, 1, W)
            nn.Flatten(2),  # (B, C, W)
        )
        
        # Interpolate to match temporal
        self.signal_length = signal_length
        
        # Final projection
        self.project = nn.Sequential(
            nn.Conv1d(256, out_dim, 1),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # x: (B, L)
        
        # Get scalogram
        scalo = self.wavelet(x)  # (B, n_scales, L)
        scalo = scalo.unsqueeze(1)  # (B, 1, n_scales, L)
        
        # 2D conv
        x = self.conv2d(scalo)
        x = self.coord_attn(x)
        x = self.conv2d_2(x)
        
        # Reduce to 1D
        x = self.reduce(x)  # (B, 256, W)
        
        # Interpolate to signal length
        x = F.interpolate(x, size=self.signal_length, mode='linear', align_corners=False)
        
        x = self.project(x)
        
        return x  # (B, out_dim, L)


class EvolutionaryFusion(nn.Module):
    """
    Evolutionary fusion of dual streams.
    
    Learns to adaptively combine temporal and spectral features
    using a gating mechanism that evolves through the network.
    """
    
    def __init__(self, dim: int):
        super().__init__()
        
        # Cross-attention between streams
        self.cross_attn = nn.MultiheadAttention(dim, num_heads=4, batch_first=True)
        
        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )
        
        # Fusion convolution
        self.fuse = nn.Sequential(
            nn.Conv1d(dim * 2, dim, 1),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True)
        )
        
        # Refinement
        self.refine = nn.Sequential(
            nn.Conv1d(dim, dim, 3, padding=1),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, temporal: torch.Tensor, spectral: torch.Tensor) -> torch.Tensor:
        # temporal, spectral: (B, C, L)
        
        # Transpose for attention
        t = temporal.transpose(1, 2)  # (B, L, C)
        s = spectral.transpose(1, 2)
        
        # Cross-attention (spectral attends to temporal)
        s_enhanced, _ = self.cross_attn(s, t, t)
        s_enhanced = s_enhanced.transpose(1, 2)  # (B, C, L)
        
        # Gating
        t_pool = temporal.mean(dim=2)  # (B, C)
        s_pool = spectral.mean(dim=2)
        gate = self.gate(torch.cat([t_pool, s_pool], dim=1))  # (B, C)
        gate = gate.unsqueeze(2)
        
        # Gated combination
        gated = gate * temporal + (1 - gate) * s_enhanced
        
        # Concatenate and fuse
        combined = torch.cat([temporal, s_enhanced], dim=1)
        fused = self.fuse(combined)
        
        # Add gated residual
        fused = fused + gated
        fused = self.refine(fused)
        
        return fused


# ============================================================================
# MAIN MODEL
# ============================================================================

class NSHT_Dual_Evo(nn.Module):
    """
    Neural Spectral-Hybrid Transformer with Dual Evolution.
    
    Architecture:
    1. Dual Input Streams:
       - Temporal Stream: 1D inception + self-attention
       - Spectral Stream: Learnable CWT + 2D conv + CoordAttention
    
    2. Evolutionary Fusion:
       - Cross-attention between streams
       - Learned gating for adaptive fusion
    
    3. Classification Head:
       - Multi-scale pooling
       - Dense layers with dropout
    
    Args:
        num_classes: Output classes (5 for AAMI)
        signal_length: Input length (360)
        hidden_dim: Hidden dimension (256)
        n_scales: Wavelet scales (32)
        dropout: Dropout rate
        use_compile: Use torch.compile()
    """
    
    def __init__(self,
                 num_classes: int = 5,
                 signal_length: int = 360,
                 hidden_dim: int = 256,
                 n_scales: int = 32,
                 dropout: float = 0.3,
                 use_compile: bool = False):
        super().__init__()
        
        self.num_classes = num_classes
        self.signal_length = signal_length
        
        # Dual streams
        self.temporal_stream = TemporalStream(out_dim=hidden_dim)
        self.spectral_stream = SpectralStream(
            n_scales=n_scales, 
            signal_length=signal_length, 
            out_dim=hidden_dim
        )
        
        # Fusion
        self.fusion = EvolutionaryFusion(hidden_dim)
        
        # Classification head
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.gmp = nn.AdaptiveMaxPool1d(1)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Feature dimension for meta-learning
        self.feature_dim = hidden_dim * 2
        
        # Initialize
        self._init_weights()
        
        if use_compile and hasattr(torch, 'compile'):
            self.forward = torch.compile(self.forward)
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, length) or (batch, 1, length)
            
        Returns:
            Logits: (batch, num_classes)
        """
        if x.dim() == 3:
            x = x.squeeze(1)
        
        # Dual stream processing
        temporal = self.temporal_stream(x)  # (B, C, L)
        spectral = self.spectral_stream(x)  # (B, C, L)
        
        # Evolutionary fusion
        fused = self.fusion(temporal, spectral)  # (B, C, L)
        
        # Global pooling
        avg_pool = self.gap(fused).squeeze(-1)
        max_pool = self.gmp(fused).squeeze(-1)
        pooled = torch.cat([avg_pool, max_pool], dim=1)
        
        # Classify
        return self.classifier(pooled)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before classifier."""
        if x.dim() == 3:
            x = x.squeeze(1)
        
        temporal = self.temporal_stream(x)
        spectral = self.spectral_stream(x)
        fused = self.fusion(temporal, spectral)
        
        avg_pool = self.gap(fused).squeeze(-1)
        max_pool = self.gmp(fused).squeeze(-1)
        
        return torch.cat([avg_pool, max_pool], dim=1)
    
    def get_stream_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get features from each stream separately.
        
        Returns:
            temporal_feat, spectral_feat, fused_feat
        """
        if x.dim() == 3:
            x = x.squeeze(1)
        
        temporal = self.temporal_stream(x)
        spectral = self.spectral_stream(x)
        fused = self.fusion(temporal, spectral)
        
        temporal_feat = self.gap(temporal).squeeze(-1)
        spectral_feat = self.gap(spectral).squeeze(-1)
        fused_feat = torch.cat([self.gap(fused).squeeze(-1), 
                                 self.gmp(fused).squeeze(-1)], dim=1)
        
        return temporal_feat, spectral_feat, fused_feat


# ============================================================================
# MODEL FACTORY
# ============================================================================

def create_nsht_dual_evo(num_classes: int = 5,
                         variant: str = 'base',
                         compile: bool = False) -> NSHT_Dual_Evo:
    """
    Factory function for NSHT variants.
    
    Args:
        num_classes: Number of classes
        variant: Model size ('tiny', 'base', 'large')
        compile: Use torch.compile()
    """
    configs = {
        'tiny': {'hidden_dim': 128, 'n_scales': 24, 'dropout': 0.2},
        'base': {'hidden_dim': 256, 'n_scales': 32, 'dropout': 0.3},
        'large': {'hidden_dim': 384, 'n_scales': 48, 'dropout': 0.4},
    }
    
    if variant not in configs:
        raise ValueError(f"Unknown variant: {variant}")
    
    return NSHT_Dual_Evo(
        num_classes=num_classes,
        use_compile=compile,
        **configs[variant]
    )


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = create_nsht_dual_evo(num_classes=5, variant='base')
    model = model.to(device)
    
    # Test
    x = torch.randn(8, 360).to(device)
    out = model(x)
    
    print(f"Model: NSHT_Dual_Evo")
    print(f"Input: {x.shape}")
    print(f"Output: {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test feature extraction
    features = model.get_features(x)
    print(f"Features: {features.shape}")
    
    # Test stream features
    t_feat, s_feat, f_feat = model.get_stream_features(x)
    print(f"Temporal features: {t_feat.shape}")
    print(f"Spectral features: {s_feat.shape}")
    print(f"Fused features: {f_feat.shape}")
