"""
ECG Arrhythmia Classification - Modular Research Codebase

Three research papers:
1. InceptionTime (1D Signal) - Context-Aware InceptionTime for ECG Classification
2. EfficientNet + Scalogram (2D Vision) - Vision Transformer Scalogram Analysis
3. NSHT-Dual-Evo (Hybrid) - Neural Spectral-Temporal Hybrid with Dual Expert Evolution

Quick Start:
    from src.data import ECGDataModule
    
    # Load pre-balanced data (SMOTE or ADASYN)
    data = ECGDataModule(mode='combined', balancing_method='smote')
    loaders = data.get_signal_loaders()
    
    # Limit samples per class for faster training
    data = ECGDataModule(mode='combined', samples_per_class=10000)
    
    # Create and train model
    from src.models import create_model
    from src.training import Trainer
    model = create_model('inception_time', num_classes=5)
    trainer = Trainer(model, loaders['train'], loaders['val'])
    trainer.train(epochs=100)
    
    # Evaluate
    from src.evaluation import evaluate_model
    results = evaluate_model(model, loaders['test'])

CLI Training:
    python scripts/train.py --config configs/paper1_inceptiontime.yaml
    python scripts/train.py --config configs/paper1_inceptiontime.yaml --kfold
"""

__version__ = "1.0.0"
__author__ = "Hardik Negi"

from . import data
from . import models
from . import training
from . import evaluation
from . import utils

__all__ = ['data', 'models', 'training', 'evaluation', 'utils', '__version__']
