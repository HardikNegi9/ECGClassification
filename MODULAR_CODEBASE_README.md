# ECG Arrhythmia Classification - Modular Research Framework

A PyTorch-based modular codebase for ECG arrhythmia classification research, supporting three research papers:

1. **Paper 1: InceptionTime** - Context-Aware InceptionTime for 1D ECG signal classification
2. **Paper 2: EfficientNet + Scalogram** - 2D vision approach using CWT scalograms with CBAM attention
3. **Paper 3: NSHT Dual-Evolution** - Neural Spectral-Hybrid Transformer with learnable wavelets

## Features

- ✅ **No Data Leakage**: SMOTE applied AFTER train/test split
- ✅ **Multi-Database Support**: MIT-BIH, INCART, and combined datasets
- ✅ **K-Fold Cross-Validation**: With proper SMOTE per fold
- ✅ **Mixed Precision Training**: AMP for faster training
- ✅ **torch.compile()**: Speedup for supported models
- ✅ **Paper-Ready Metrics**: Per-class precision, recall, specificity, F1, AUC-ROC
- ✅ **Comprehensive Visualizations**: Confusion matrices, ROC curves, learning curves
- ✅ **Experiment Tracking**: Automatic logging, checkpointing, and results saving

## Project Structure

```
ClassficationECG/
├── src/                          # Main source package
│   ├── data/                     # Data loading and preprocessing
│   │   ├── dataset.py           # ECGDataModule with SMOTE handling
│   │   └── download.py          # PhysioNet download utilities
│   ├── models/                   # Model architectures
│   │   ├── inception_time.py    # Paper 1: Context-Aware InceptionTime
│   │   ├── efficientnet_scalogram.py  # Paper 2: EfficientNet + CBAM
│   │   └── nsht_dual_evo.py     # Paper 3: NSHT Dual-Evolution
│   ├── training/                 # Training utilities
│   │   └── trainer.py           # Trainer and KFoldTrainer
│   ├── evaluation/               # Metrics and evaluation
│   │   └── metrics.py           # PaperMetrics, visualizations
│   └── utils/                    # Configuration and utilities
│       └── config.py            # ExperimentConfig, tracking
├── configs/                      # YAML configuration files
│   ├── paper1_inceptiontime.yaml
│   ├── paper2_efficientnet.yaml
│   └── paper3_nsht.yaml
├── scripts/                      # CLI entry points
│   ├── train.py                 # Main training script
│   ├── evaluate.py              # Evaluation script
│   └── train_all.py             # Train all models
├── balanced_data/                # Processed datasets
├── experiments/                  # Experiment outputs
├── checkpoints/                  # Model checkpoints
└── requirements.txt
```

## Installation

```bash
# Clone and navigate to project
cd ClassficationECG

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install PyTorch with CUDA (adjust for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install other requirements
pip install -r requirements.txt
```

## Quick Start

### 1. Download and Prepare Data

```bash
# Download MIT-BIH and INCART databases
python -m src.data.download --download-all

# Create RAW datasets (no SMOTE - prevents leakage)
python -m src.data.download --create-raw
```

### 2. Train a Model

```bash
# Train InceptionTime (Paper 1)
python scripts/train.py --config configs/paper1_inceptiontime.yaml

# Train EfficientNet (Paper 2)
python scripts/train.py --config configs/paper2_efficientnet.yaml

# Train NSHT (Paper 3)
python scripts/train.py --config configs/paper3_nsht.yaml
```

### 3. K-Fold Cross-Validation (for paper results)

```bash
# Run 10-fold CV with proper SMOTE handling
python scripts/train.py --config configs/paper1_inceptiontime.yaml --kfold
```

### 4. Evaluate a Model

```bash
python scripts/evaluate.py \
    --model-path checkpoints/paper1_inceptiontime/best_model.pt \
    --config configs/paper1_inceptiontime.yaml \
    --benchmark --latex
```

## Python API Usage

```python
from src.data import ECGDataModule
from src.models import create_model
from src.training import Trainer
from src.evaluation import evaluate_model, PaperMetrics

# Load pre-balanced data (SMOTE or ADASYN)
data = ECGDataModule(mode='combined', balancing_method='smote')
loaders = data.get_signal_loaders(batch_size=2048)

# Limit samples per class for faster training (e.g., 10k per class = 50k total)
data = ECGDataModule(mode='combined', samples_per_class=10000)

# Create model
model = create_model('inception_time', num_classes=5, variant='base')
model = model.cuda()

# Train
trainer = Trainer(
    model=model,
    train_loader=loaders['train'],
    val_loader=loaders['val'],
    use_amp=True,
    use_class_weights=True
)
history = trainer.train(epochs=100, patience=15)

# Evaluate
results = evaluate_model(model, loaders['test'])
metrics = results['metrics']
metrics.print_full_report()
```

## Configuration

Edit YAML configs in `configs/` or override via command line:

```bash
python scripts/train.py --config configs/paper1_inceptiontime.yaml \
    --model.variant large \
    --training.epochs 150 \
    --training.lr 0.0005 \
    --data.mode mitbih
```

## Data Modes

| Mode | Description |
|------|-------------|
| `combined` | MIT-BIH + INCART combined |
| `mitbih` | MIT-BIH Arrhythmia Database only |
| `incart` | INCART Database only |
| `cross_mit_incart` | Train on MIT-BIH, test on INCART |
| `cross_incart_mit` | Train on INCART, test on MIT-BIH |

## AAMI Classes

| Class | Label | Description |
|-------|-------|-------------|
| 0 | N | Normal beats |
| 1 | S | Supraventricular ectopic beats |
| 2 | V | Ventricular ectopic beats |
| 3 | F | Fusion beats |
| 4 | Q | Unknown/Paced beats |

## Configuration Options

Choose between SMOTE and ADASYN pre-balanced datasets:

```python
# SMOTE-balanced (default)
data = ECGDataModule(balancing_method='smote')

# ADASYN-balanced
data = ECGDataModule(balancing_method='adasyn')

# Limit samples per class
data = ECGDataModule(samples_per_class=10000)  # 10k per class = 50k total
```

## Citation

If you use this code, please cite:

```bibtex
@software{ecg_classification_2024,
  title={ECG Arrhythmia Classification: A Modular Research Framework},
  author={ECG Research Team},
  year={2024},
  url={https://github.com/your-repo/ecg-classification}
}
```

## License

MIT License
