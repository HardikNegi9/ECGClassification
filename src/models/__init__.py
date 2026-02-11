"""
ECG Classification Models

Paper 1: InceptionTime (1D Signal Processing)
Paper 2: EfficientNet + Scalogram (2D Vision)
Paper 3: NSHT Dual-Evolution (Hybrid)
"""

from .inception_time import (
    ContextAwareInceptionTime,
    create_inception_time,
    InceptionModule,
    DilatedInceptionModule,
    SEBlock
)

from .efficientnet_scalogram import (
    AttentionEfficientNet,
    create_efficientnet_scalogram,
    CWTScalogramLayer,
    CBAM,
    ChannelAttention,
    SpatialAttention
)

from .nsht_dual_evo import (
    NSHT_Dual_Evo,
    create_nsht_dual_evo,
    AdaptiveMorletWavelet,
    CoordinateAttention,
    TemporalStream,
    SpectralStream,
    EvolutionaryFusion
)

from .dual_ensemble import (
    DualEnsemble,
    create_dual_ensemble,
    ensemble_inference,
    WeightedFusion,
    LearnableFusion,
    AttentionFusion,
    ScalogramConverter
)


__all__ = [
    # Paper 1
    'ContextAwareInceptionTime',
    'create_inception_time',
    'InceptionModule',
    'DilatedInceptionModule',
    'SEBlock',
    
    # Paper 2
    'AttentionEfficientNet',
    'create_efficientnet_scalogram',
    'CWTScalogramLayer',
    'CBAM',
    'ChannelAttention',
    'SpatialAttention',
    
    # Paper 3
    'NSHT_Dual_Evo',
    'create_nsht_dual_evo',
    'AdaptiveMorletWavelet',
    'CoordinateAttention',
    'TemporalStream',
    'SpectralStream',
    'EvolutionaryFusion',
    
    # Ensemble
    'DualEnsemble',
    'create_dual_ensemble',
    'ensemble_inference',
    'WeightedFusion',
    'LearnableFusion',
    'AttentionFusion',
    'ScalogramConverter',
]


# Model registry for easy access
MODEL_REGISTRY = {
    'inception_time': create_inception_time,
    'efficientnet_scalogram': create_efficientnet_scalogram,
    'nsht_dual_evo': create_nsht_dual_evo,
    'dual_ensemble': create_dual_ensemble,
}


def create_model(name: str, num_classes: int = 5, **kwargs):
    """
    Factory function to create any model by name.
    
    Args:
        name: Model name ('inception_time', 'efficientnet_scalogram', 'nsht_dual_evo')
        num_classes: Number of output classes
        **kwargs: Additional model-specific arguments
        
    Returns:
        Configured model
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODEL_REGISTRY.keys())}")
    
    return MODEL_REGISTRY[name](num_classes=num_classes, **kwargs)


def list_models():
    """List available model names."""
    return list(MODEL_REGISTRY.keys())
