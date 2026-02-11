"""
InceptionTime Model - Paper 1 (1D Signal Classification)

Context-Aware InceptionTime with:
- Dilated inception branches (multi-scale temporal features)
- SE-style attention
- Residual connections
- Context module for R-peak signal understanding

Reference: 
- Fawaz et al., "InceptionTime: Finding AlexNet for Time Series Classification"

Usage:
    from src.models import ContextAwareInceptionTime
    
    model = ContextAwareInceptionTime(num_classes=5)
    out = model(x)  # x: (batch, 360)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ============================================================================
# BUILDING BLOCKS
# ============================================================================

class SEBlock(nn.Module):
    """Squeeze-and-Excitation for 1D signals."""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excite = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x: (B, C, L)
        b, c, _ = x.shape
        y = self.squeeze(x).view(b, c)
        y = self.excite(y).view(b, c, 1)
        return x * y.expand_as(x)


class InceptionModule(nn.Module):
    """
    Inception module with bottleneck and multi-scale kernels.
    """
    
    def __init__(self, in_channels: int, n_filters: int = 32, 
                 kernel_sizes: tuple = (9, 19, 39), bottleneck: int = 32,
                 use_se: bool = True):
        super().__init__()
        
        self.use_se = use_se
        
        # Bottleneck conv
        self.bottleneck = nn.Conv1d(in_channels, bottleneck, 1, bias=False)
        
        # Multi-scale convolutions
        self.convs = nn.ModuleList([
            nn.Conv1d(bottleneck, n_filters, k, padding=k//2, bias=False)
            for k in kernel_sizes
        ])
        
        # MaxPool branch
        self.maxpool = nn.MaxPool1d(3, stride=1, padding=1)
        self.pool_conv = nn.Conv1d(in_channels, n_filters, 1, bias=False)
        
        # Output channels = len(kernel_sizes) + 1 branches
        out_channels = n_filters * (len(kernel_sizes) + 1)
        
        # BatchNorm and activation
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU(inplace=True)
        
        # SE attention
        if use_se:
            self.se = SEBlock(out_channels)
    
    def forward(self, x):
        # Bottleneck
        x_bn = self.bottleneck(x)
        
        # Multi-scale convolutions
        outputs = [conv(x_bn) for conv in self.convs]
        
        # MaxPool branch
        outputs.append(self.pool_conv(self.maxpool(x)))
        
        # Concatenate
        out = torch.cat(outputs, dim=1)
        out = self.act(self.bn(out))
        
        if self.use_se:
            out = self.se(out)
        
        return out


class DilatedInceptionModule(nn.Module):
    """
    Inception module with dilated convolutions for larger receptive field.
    """
    
    def __init__(self, in_channels: int, n_filters: int = 32,
                 kernel_size: int = 15, dilations: tuple = (1, 2, 4, 8)):
        super().__init__()
        
        self.bottleneck = nn.Conv1d(in_channels, n_filters, 1, bias=False)
        
        self.dilated_convs = nn.ModuleList([
            nn.Conv1d(n_filters, n_filters, kernel_size, 
                     padding=(kernel_size//2)*d, dilation=d, bias=False)
            for d in dilations
        ])
        
        out_channels = n_filters * len(dilations)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.se = SEBlock(out_channels)
    
    def forward(self, x):
        x = self.bottleneck(x)
        outputs = [conv(x) for conv in self.dilated_convs]
        out = torch.cat(outputs, dim=1)
        out = self.se(self.act(self.bn(out)))
        return out


class ResidualBlock(nn.Module):
    """Residual connection wrapper."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.shortcut = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels)
        ) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x, y):
        return F.relu(self.shortcut(x) + y)


class ContextModule(nn.Module):
    """
    Context enhancement for R-peak centered signals.
    
    Enhances features around the R-peak (center) while maintaining
    context from PR and ST segments.
    """
    
    def __init__(self, channels: int):
        super().__init__()
        
        # Separate projections for P-wave, QRS, T-wave regions
        self.region_attn = nn.Sequential(
            nn.Conv1d(channels, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels // 4, 3, 1),  # 3 regions
            nn.Softmax(dim=-1)
        )
        
        self.fuse = nn.Conv1d(channels * 3, channels, 1)
    
    def forward(self, x):
        B, C, L = x.shape
        
        # Split into 3 regions (P-wave, QRS, T-wave for RPeak signals)
        third = L // 3
        regions = [
            x[:, :, :third],           # Before R-peak (P-wave region)
            x[:, :, third:2*third],    # Around R-peak (QRS)
            x[:, :, 2*third:]          # After R-peak (T-wave region)
        ]
        
        # Global features for each region
        region_feats = [r.mean(dim=2, keepdim=True).expand(-1, -1, r.size(2)) 
                        for r in regions]
        
        # Concatenate back
        context = torch.cat([
            torch.cat([region_feats[0], torch.zeros(B, C, L-third, device=x.device)], dim=2),
            torch.cat([torch.zeros(B, C, third, device=x.device), 
                       region_feats[1], 
                       torch.zeros(B, C, L-2*third, device=x.device)], dim=2),
            torch.cat([torch.zeros(B, C, 2*third, device=x.device), region_feats[2]], dim=2)
        ], dim=1)
        
        return x + self.fuse(context)


# ============================================================================
# MAIN MODEL
# ============================================================================

class ContextAwareInceptionTime(nn.Module):
    """
    Context-Aware InceptionTime for ECG Classification.
    
    Features:
    - Input embedding with position encoding
    - Stacked inception modules with residual connections
    - Dilated inception for multi-scale features
    - Context module for R-peak signal understanding
    - SE attention throughout
    
    Args:
        num_classes: Number of output classes (5 for AAMI)
        input_length: Signal length (360 for 1sec @ 360Hz)
        n_filters: Base filter count (32)
        depth: Number of inception blocks (6)
        use_compile: Use torch.compile() for speedup
    """
    
    def __init__(self, 
                 num_classes: int = 5,
                 input_length: int = 360,
                 n_filters: int = 32,
                 depth: int = 6,
                 dropout: float = 0.3,
                 use_compile: bool = False):
        super().__init__()
        
        self.num_classes = num_classes
        self.input_length = input_length
        
        # Input embedding (1 channel -> n_filters channels)
        self.input_embed = nn.Sequential(
            nn.Conv1d(1, n_filters, 7, padding=3, bias=False),
            nn.BatchNorm1d(n_filters),
            nn.ReLU(inplace=True)
        )
        
        # Learnable position encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, n_filters, input_length) * 0.02)
        
        # Build network layers
        layers = []
        residuals = []
        
        current_channels = n_filters
        
        for i in range(depth):
            # Alternate between standard and dilated inception
            if i % 2 == 0:
                out_ch = n_filters * 4  # 4 branches
                module = InceptionModule(current_channels, n_filters)
            else:
                out_ch = n_filters * 4  # 4 dilations
                module = DilatedInceptionModule(current_channels, n_filters)
            
            layers.append(module)
            residuals.append(ResidualBlock(current_channels, out_ch))
            current_channels = out_ch
        
        self.layers = nn.ModuleList(layers)
        self.residuals = nn.ModuleList(residuals)
        
        # Context module
        self.context = ContextModule(current_channels)
        
        # Classification head
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.gmp = nn.AdaptiveMaxPool1d(1)
        
        self.classifier = nn.Sequential(
            nn.Linear(current_channels * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
        
        # Compile if requested
        if use_compile and hasattr(torch, 'compile'):
            self.forward = torch.compile(self.forward)
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, length) or (batch, 1, length)
            
        Returns:
            Logits of shape (batch, num_classes)
        """
        # Handle input shape
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, L) -> (B, 1, L)
        
        # Input embedding
        x = self.input_embed(x)
        
        # Add position encoding
        x = x + self.pos_encoding
        
        # Inception blocks with residual
        for layer, residual in zip(self.layers, self.residuals):
            identity = x
            x = layer(x)
            x = residual(identity, x)
        
        # Context enhancement
        x = self.context(x)
        
        # Global pooling (concat avg + max)
        avg_pool = self.gap(x).squeeze(-1)
        max_pool = self.gmp(x).squeeze(-1)
        pooled = torch.cat([avg_pool, max_pool], dim=1)
        
        # Classify
        return self.classifier(pooled)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before classifier (for ensemble/meta-learning)."""
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        x = self.input_embed(x)
        x = x + self.pos_encoding
        
        for layer, residual in zip(self.layers, self.residuals):
            identity = x
            x = layer(x)
            x = residual(identity, x)
        
        x = self.context(x)
        
        avg_pool = self.gap(x).squeeze(-1)
        max_pool = self.gmp(x).squeeze(-1)
        return torch.cat([avg_pool, max_pool], dim=1)


# ============================================================================
# MODEL FACTORY
# ============================================================================

def create_inception_time(num_classes: int = 5,
                          variant: str = 'base',
                          compile: bool = False) -> ContextAwareInceptionTime:
    """
    Factory function to create InceptionTime variants.
    
    Args:
        num_classes: Number of output classes
        variant: Model size ('tiny', 'base', 'large')
        compile: Use torch.compile()
    
    Returns:
        Configured model
    """
    configs = {
        'tiny': {'n_filters': 16, 'depth': 4, 'dropout': 0.2},
        'base': {'n_filters': 32, 'depth': 6, 'dropout': 0.3},
        'large': {'n_filters': 64, 'depth': 8, 'dropout': 0.4},
    }
    
    if variant not in configs:
        raise ValueError(f"Unknown variant: {variant}. Choose from {list(configs.keys())}")
    
    return ContextAwareInceptionTime(
        num_classes=num_classes,
        use_compile=compile,
        **configs[variant]
    )


if __name__ == "__main__":
    # Quick test
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = create_inception_time(num_classes=5, variant='base')
    model = model.to(device)
    
    # Test forward pass
    x = torch.randn(8, 360).to(device)
    out = model(x)
    
    print(f"Model: ContextAwareInceptionTime")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test feature extraction
    features = model.get_features(x)
    print(f"Features shape: {features.shape}")
