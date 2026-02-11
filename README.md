# ECG Arrhythmia Classification - Modular Research Framework

A PyTorch-based modular codebase for ECG arrhythmia classification research, supporting three research papers plus an ensemble approach:

1. **Paper 1: InceptionTime** - Context-Aware InceptionTime for 1D ECG signal classification
2. **Paper 2: EfficientNet + Scalogram** - 2D vision approach using CWT scalograms with CBAM attention
3. **Paper 3: NSHT Dual-Evolution** - Neural Spectral-Hybrid Transformer with learnable wavelets
4. **Dual Ensemble** - Signal + Vision fusion combining InceptionTime and EfficientNet experts

## Features

- ✅ **Pre-Balanced Data**: Uses SMOTE/ADASYN pre-balanced datasets for consistent class distribution
- ✅ **Multi-Database Support**: MIT-BIH, INCART, and combined datasets
- ✅ **K-Fold Cross-Validation**: Stratified splits for robust evaluation
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
│   │   └── download.py          # Download, create RAW, and balance datasets
│   ├── models/                   # Model architectures
│   │   ├── inception_time.py    # Paper 1: Context-Aware InceptionTime
│   │   ├── efficientnet_scalogram.py  # Paper 2: EfficientNet + CBAM
│   │   ├── nsht_dual_evo.py     # Paper 3: NSHT Dual-Evolution
│   │   └── dual_ensemble.py     # Dual Ensemble: Signal + Vision Fusion
│   ├── training/                 # Training utilities
│   │   └── trainer.py           # Trainer and KFoldTrainer
│   ├── evaluation/               # Metrics and evaluation
│   │   └── metrics.py           # PaperMetrics, visualizations
│   └── utils/                    # Configuration and utilities
│       └── config.py            # ExperimentConfig, tracking
├── configs/                      # YAML configuration files
│   ├── paper1_inceptiontime.yaml
│   ├── paper2_efficientnet.yaml
│   ├── paper3_nsht.yaml
│   └── dual_ensemble.yaml       # Dual Ensemble configuration
├── scripts/                      # CLI entry points
│   ├── train.py                 # Main training script
│   ├── evaluate.py              # Evaluation script
│   ├── eval_dual_ensemble.py    # Dual Ensemble evaluation script
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

# Create all RAW datasets (MIT-BIH, INCART, Combined)
python -m src.data.download --create-raw

# Create specific dataset only
python -m src.data.download --create-raw --mode mitbih
python -m src.data.download --create-raw --mode incart
python -m src.data.download --create-raw --mode combined
```

### 2. Create SMOTE-Balanced Datasets (Required for Training)

```bash
# Create balanced dataset from combined RAW data (default: 15,000 samples/class)
python -m src.data.download --create-balanced --mode combined

# Create balanced dataset from MIT-BIH only
python -m src.data.download --create-balanced --mode mitbih

# Create balanced dataset from INCART only
python -m src.data.download --create-balanced --mode incart

# Create all balanced datasets (MIT-BIH, INCART, Combined)
python -m src.data.download --create-balanced --mode all

# Custom samples per class
python -m src.data.download --create-balanced --mode combined --samples-per-class 20000
```

**Options:**
| Flag | Description |
|------|-------------|
| `--mode` | `mitbih`, `incart`, `combined`, or `all` (default: all) |
| `--samples-per-class` | Target samples per class after SMOTE (default: 15000) |

This creates:
- `balanced_data/X_balanced_rpeak_smote.npy` (for combined)
- `balanced_data/y_balanced_rpeak_smote.npy`
- `balanced_data/X_mitbih_rpeak_smote.npy` (for mitbih)
- `balanced_data/X_incart_rpeak_smote.npy` (for incart)

**Note:** Requires RAW datasets to exist first (`--create-raw`).

**Alternative:** Use standalone script for more options:
```bash
python create_balanced_rpeak_segments.py --datasets both --samples-per-class 15000 --no-visual
```

**Runtime sampling** can further limit data via `ECGDataModule`:

```python
from src.data import ECGDataModule

# Load all samples
data = ECGDataModule(mode='combined', balancing_method='smote')

# Limit to 10,000 samples per class (50k total for 5 classes)
data = ECGDataModule(mode='combined', samples_per_class=10000)
```

### 3. Train a Model

```bash
# Train InceptionTime (Paper 1)
python scripts/train.py --config configs/paper1_inceptiontime.yaml

# Train EfficientNet (Paper 2)
python scripts/train.py --config configs/paper2_efficientnet.yaml

# Train NSHT (Paper 3)
python scripts/train.py --config configs/paper3_nsht.yaml

# Evaluate Dual Ensemble (uses pre-trained Signal + Vision experts)
python scripts/eval_dual_ensemble.py --config configs/dual_ensemble.yaml
```

### 4. K-Fold Cross-Validation (for paper results)

```bash
# Run 10-fold CV with proper SMOTE handling
python scripts/train.py --config configs/paper1_inceptiontime.yaml --kfold
```

### 5. Evaluate a Model

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

### Dual Ensemble Usage

```python
from src.models import DualEnsemble, create_dual_ensemble, ensemble_inference

# Option 1: Create from pre-trained experts
ensemble = DualEnsemble.from_pretrained(
    signal_path='models/signal_expert_best.pth',
    vision_path='models/vision_expert_best.pth',
    fusion='weighted',  # Options: 'weighted', 'learnable', 'attention', 'feature'
    freeze_experts=True
)

# Option 2: Use factory function
ensemble = create_dual_ensemble(
    num_classes=5,
    fusion='weighted',
    signal_weight=0.60,
    vision_weight=0.40,
    signal_path='models/signal_expert_best.pth',
    vision_path='models/vision_expert_best.pth'
)

# Inference
predictions, probabilities = ensemble.predict(X_test, batch_size=256)

# Or use standalone inference function with separate models
result = ensemble_inference(
    signal_model=signal_expert,
    vision_model=vision_expert,
    X=X_test,
    signal_weight=0.6,
    vision_weight=0.4
)
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

## Model Registry

Use the factory function to create any model by name:

```python
from src.models import create_model, list_models

# Available models
print(list_models())  # ['inception_time', 'efficientnet_scalogram', 'nsht_dual_evo', 'dual_ensemble']

# Create any model
model = create_model('inception_time', num_classes=5, variant='base')
model = create_model('efficientnet_scalogram', num_classes=5, variant='b0')
model = create_model('nsht_dual_evo', num_classes=5)
model = create_model('dual_ensemble', num_classes=5, fusion='weighted')
```

| Model | Description | Key Args |
|-------|-------------|----------|
| `inception_time` | Context-Aware InceptionTime (1D) | `variant`: 'tiny', 'base', 'large' |
| `efficientnet_scalogram` | EfficientNet + CBAM (2D scalograms) | `variant`: 'b0', 'b1', etc. |
| `nsht_dual_evo` | NSHT Dual-Evolution Transformer | - |
| `dual_ensemble` | Signal + Vision Fusion | `fusion`: 'weighted', 'learnable', 'attention', 'feature' |

## Fusion Strategies (Dual Ensemble)

| Strategy | Description |
|----------|-------------|
| `weighted` | Simple weighted average (default: 60% signal, 40% vision) |
| `learnable` | MLP on concatenated logits |
| `attention` | Attention network for dynamic per-sample weighting |
| `feature` | Fusion on feature level before classification heads |

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
@software{ecg_classification_2026,
  title={ECG Arrhythmia Classification: A Modular Research Framework},
  author={ECG Research Team},
  year={2026},
  url={https://github.com/your-repo/ecg-classification}
}
```

## License

MIT License
