"""
Utility Functions and Configuration Loading

Includes:
- YAML config loading/parsing
- Logging setup
- Seed setting
- Device detection
- Experiment tracking
"""

import os
import sys
import yaml
import json
import random
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field, asdict

import numpy as np
import torch


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class DataConfig:
    """Data configuration."""
    mode: str = 'combined'  # combined, mitbih, incart, cross_mit_incart, cross_incart_mit
    balancing_method: str = 'smote'  # 'smote' or 'adasyn'
    samples_per_class: int = None  # Samples per class (null = use all, e.g., 10000 = 50k total)
    test_size: float = 0.20
    val_size: float = 0.15
    batch_size: int = 2048
    num_workers: int = 4
    seed: int = 42


@dataclass
class ModelConfig:
    """Model configuration."""
    name: str = 'inception_time'  # inception_time, efficientnet_scalogram, nsht_dual_evo
    variant: str = 'base'  # tiny, base, large (or b0-b7 for efficientnet)
    num_classes: int = 5
    dropout: float = 0.3
    use_compile: bool = False
    
    # EfficientNet-specific
    pretrained: bool = True
    img_size: int = 224
    cbam_reduction: int = 16
    
    # NSHT-specific
    hidden_dim: int = 256
    n_scales: int = 32


@dataclass
class TrainingConfig:
    """Training configuration."""
    epochs: int = 100
    patience: int = 10  # Epochs to wait after all LR reductions
    lr_patience: int = 5  # Epochs to wait before reducing LR
    warmup_epochs: int = 5
    lr: float = 1e-3
    lr_reduce_factor: float = 0.5  # Factor to reduce LR
    min_lr: float = 1e-7  # Minimum learning rate
    max_lr_reductions: int = 3  # Max LR reductions before stopping
    weight_decay: float = 1e-4
    use_amp: bool = True
    use_class_weights: bool = True
    gradient_clip: float = 1.0


@dataclass
class KFoldConfig:
    """K-Fold configuration."""
    enabled: bool = False
    n_splits: int = 10
    apply_smote_per_fold: bool = True
    smote_target: int = None  # Target samples per class (None = auto)


@dataclass
class ExperimentConfig:
    """Full experiment configuration."""
    name: str = 'experiment'
    seed: int = 42
    device: str = 'auto'  # auto, cuda, cuda:0, cpu
    
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    kfold: KFoldConfig = field(default_factory=KFoldConfig)
    
    # Paths
    checkpoint_dir: str = 'checkpoints'
    experiment_dir: str = 'experiments'
    log_dir: str = 'logs'
    
    def __post_init__(self):
        # Convert nested dicts to dataclasses if needed
        if isinstance(self.data, dict):
            self.data = DataConfig(**self.data)
        if isinstance(self.model, dict):
            self.model = ModelConfig(**self.model)
        if isinstance(self.training, dict):
            self.training = TrainingConfig(**self.training)
        if isinstance(self.kfold, dict):
            self.kfold = KFoldConfig(**self.kfold)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def save(self, path: str):
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    @classmethod
    def load(cls, path: str) -> 'ExperimentConfig':
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)


def load_config(path: str) -> ExperimentConfig:
    """Load configuration from YAML file."""
    return ExperimentConfig.load(path)


def create_default_config() -> ExperimentConfig:
    """Create default configuration."""
    return ExperimentConfig()


# ============================================================================
# DEVICE & SEED
# ============================================================================

def get_device(device_str: str = 'auto') -> torch.device:
    """
    Get torch device.
    
    Args:
        device_str: 'auto', 'cuda', 'cuda:0', 'cuda:1', 'cpu'
        
    Returns:
        torch.device
    """
    if device_str == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')
    return torch.device(device_str)


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_gpu_info() -> Dict[str, Any]:
    """Get GPU information."""
    info = {
        'cuda_available': torch.cuda.is_available(),
        'device_count': 0,
        'devices': [],
    }
    
    if torch.cuda.is_available():
        info['device_count'] = torch.cuda.device_count()
        for i in range(info['device_count']):
            props = torch.cuda.get_device_properties(i)
            info['devices'].append({
                'name': props.name,
                'total_memory_gb': props.total_memory / 1e9,
                'compute_capability': f'{props.major}.{props.minor}',
            })
    
    return info


# ============================================================================
# LOGGING
# ============================================================================

def setup_logging(log_dir: str = 'logs', 
                  experiment_name: str = None,
                  level: int = logging.INFO) -> logging.Logger:
    """
    Setup logging for experiments.
    
    Args:
        log_dir: Directory for log files
        experiment_name: Name for log file
        level: Logging level
        
    Returns:
        Logger instance
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    name = experiment_name or 'experiment'
    log_file = log_dir / f'{name}_{timestamp}.log'
    
    # Create logger
    logger = logging.getLogger('ecg_classification')
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers = []
    
    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(level)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


# ============================================================================
# EXPERIMENT TRACKING
# ============================================================================

class ExperimentTracker:
    """
    Simple experiment tracking for results and metrics.
    """
    
    def __init__(self, experiment_name: str, base_dir: str = 'experiments'):
        self.name = experiment_name
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.dir = Path(base_dir) / f'{experiment_name}_{self.timestamp}'
        self.dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics = {}
        self.artifacts = []
    
    def log_config(self, config: Union[ExperimentConfig, Dict]):
        """Save configuration."""
        if isinstance(config, ExperimentConfig):
            config = config.to_dict()
        
        with open(self.dir / 'config.yaml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    
    def log_metric(self, name: str, value: float, step: int = None):
        """Log a metric value."""
        if name not in self.metrics:
            self.metrics[name] = []
        
        self.metrics[name].append({
            'value': value,
            'step': step,
            'timestamp': datetime.now().isoformat(),
        })
    
    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """Log multiple metrics."""
        for name, value in metrics.items():
            self.log_metric(name, value, step)
    
    def save_model(self, model: torch.nn.Module, name: str = 'model.pt'):
        """Save model weights."""
        path = self.dir / name
        torch.save(model.state_dict(), path)
        self.artifacts.append(str(path))
    
    def save_results(self, results: Dict[str, Any], name: str = 'results.json'):
        """Save results to JSON."""
        path = self.dir / name
        
        # Convert numpy arrays to lists
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj
        
        results = convert(results)
        
        with open(path, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.artifacts.append(str(path))
    
    def save_figure(self, fig, name: str):
        """Save matplotlib figure."""
        path = self.dir / name
        fig.savefig(path, dpi=300, bbox_inches='tight')
        self.artifacts.append(str(path))
    
    def finalize(self):
        """Save all metrics and create summary."""
        # Save metrics history
        with open(self.dir / 'metrics.json', 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        # Create summary
        summary = {
            'name': self.name,
            'timestamp': self.timestamp,
            'artifacts': self.artifacts,
            'final_metrics': {
                name: values[-1]['value'] if values else None
                for name, values in self.metrics.items()
            }
        }
        
        with open(self.dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nExperiment saved to: {self.dir}")


# ============================================================================
# MISC UTILITIES
# ============================================================================

def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model: torch.nn.Module, input_shape: tuple = (1, 360)):
    """Print model summary."""
    print(f"\n{'='*60}")
    print(f"Model: {model.__class__.__name__}")
    print(f"{'='*60}")
    print(f"Trainable Parameters: {count_parameters(model):,}")
    
    # Try forward pass (use eval mode to avoid BatchNorm issues with batch_size=1)
    try:
        was_training = model.training
        model.eval()
        device = next(model.parameters()).device
        # Use batch_size=2 to avoid BatchNorm issues
        batch_shape = (2,) + input_shape if len(input_shape) == 1 else (2,) + input_shape[1:]
        x = torch.randn(*batch_shape).to(device)
        with torch.no_grad():
            out = model(x)
        print(f"Input Shape: {batch_shape}")
        print(f"Output Shape: {tuple(out.shape)}")
        if was_training:
            model.train()
    except Exception as e:
        print(f"Forward pass test failed: {e}")
    
    print('='*60 + '\n')


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Config
    'DataConfig',
    'ModelConfig', 
    'TrainingConfig',
    'KFoldConfig',
    'ExperimentConfig',
    'load_config',
    'create_default_config',
    
    # Device & Seed
    'get_device',
    'set_seed',
    'get_gpu_info',
    
    # Logging
    'setup_logging',
    
    # Tracking
    'ExperimentTracker',
    
    # Utils
    'count_parameters',
    'print_model_summary',
]
