"""
Dual Ensemble Evaluation Script

Runs evaluation on pre-trained Signal + Vision experts with weighted fusion.

Usage:
    python scripts/eval_dual_ensemble.py
    python scripts/eval_dual_ensemble.py --config configs/dual_ensemble.yaml
    python scripts/eval_dual_ensemble.py --signal_weight 0.55 --vision_weight 0.45
"""

import os
import sys
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import (
    ContextAwareInceptionTime, 
    AttentionEfficientNet,
    ScalogramConverter
)


def clean_state_dict(state_dict):
    """Remove '_orig_mod.' prefix from torch.compile state dicts."""
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            k = k[10:]
        new_state_dict[k] = v
    return new_state_dict


class ScalogramDataset(Dataset):
    """Dataset for creating scalograms on-the-fly."""
    
    def __init__(self, signals, img_size=224):
        self.signals = signals
        self.converter = ScalogramConverter(img_size=img_size)
    
    def __len__(self):
        return len(self.signals)
    
    def __getitem__(self, idx):
        return self.converter(self.signals[idx])


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_data(config: dict):
    """Load test data."""
    print("\n[1/5] Loading Data...")
    
    signals_path = config['data']['raw_signals_path']
    labels_path = config['data']['labels_path']
    test_idx_path = config['data']['test_indices_path']
    
    X = np.load(signals_path)
    y = np.load(labels_path).astype(int)
    test_idx = np.load(test_idx_path)
    
    X_test = X[test_idx]
    y_test = y[test_idx]
    
    print(f"   Loaded {len(y_test)} test samples")
    print(f"   Signal shape: {X_test.shape}")
    print(f"   Classes: {np.unique(y_test)}")
    
    return X_test, y_test


def load_models(config: dict, device: torch.device):
    """Load pre-trained Signal and Vision experts."""
    print("\n[2/5] Loading Models...")
    
    num_classes = config['data']['num_classes']
    
    # Signal Expert
    signal_path = config['models']['signal_expert_path']
    signal_model = ContextAwareInceptionTime(n_classes=num_classes).to(device)
    
    if os.path.exists(signal_path):
        state_dict = torch.load(signal_path, map_location=device)
        signal_model.load_state_dict(clean_state_dict(state_dict))
        print(f"   ✓ Signal Expert loaded from: {signal_path}")
    else:
        raise FileNotFoundError(f"Signal Expert not found: {signal_path}")
    
    # Vision Expert
    vision_path = config['models']['vision_expert_path']
    vision_model = AttentionEfficientNet(n_classes=num_classes).to(device)
    
    if os.path.exists(vision_path):
        state_dict = torch.load(vision_path, map_location=device)
        vision_model.load_state_dict(clean_state_dict(state_dict))
        print(f"   ✓ Vision Expert loaded from: {vision_path}")
    else:
        raise FileNotFoundError(f"Vision Expert not found: {vision_path}")
    
    signal_model.eval()
    vision_model.eval()
    
    return signal_model, vision_model


@torch.no_grad()
def run_signal_expert(model, signals, batch_size, device):
    """Run inference with Signal Expert."""
    print("\n[3/5] Running Signal Expert...")
    
    ds = TensorDataset(torch.FloatTensor(signals))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    
    probs = []
    for (xb,) in tqdm(dl, desc="Signal Expert", ncols=100):
        out = model(xb.to(device))
        probs.append(F.softmax(out, dim=1).cpu().numpy())
    
    probs = np.vstack(probs)
    preds = probs.argmax(axis=1)
    
    return probs, preds


@torch.no_grad()
def run_vision_expert(model, signals, batch_size, img_size, device):
    """Run inference with Vision Expert."""
    print("\n[4/5] Running Vision Expert...")
    
    ds = ScalogramDataset(signals, img_size=img_size)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    
    probs = []
    for xb in tqdm(dl, desc="Vision Expert", ncols=100):
        out = model(xb.to(device))
        probs.append(F.softmax(out, dim=1).cpu().numpy())
    
    probs = np.vstack(probs)
    preds = probs.argmax(axis=1)
    
    return probs, preds


def fuse_predictions(probs_signal, probs_vision, signal_weight, vision_weight):
    """Fuse predictions with weighted average."""
    print("\n[5/5] Fusing Predictions...")
    print(f"   Signal weight: {signal_weight:.2f}")
    print(f"   Vision weight: {vision_weight:.2f}")
    
    final_probs = probs_signal * signal_weight + probs_vision * vision_weight
    final_preds = final_probs.argmax(axis=1)
    
    return final_probs, final_preds


def compute_metrics(y_true, y_pred, class_names):
    """Compute and print metrics."""
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    cm = confusion_matrix(y_true, y_pred)
    
    return {
        'accuracy': acc,
        'report': report,
        'confusion_matrix': cm
    }


def plot_confusion_matrix(cm, class_names, save_path=None):
    """Plot confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Dual Ensemble Confusion Matrix')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"   Saved: {save_path}")
    plt.close()


def plot_weight_analysis(probs_signal, probs_vision, y_true, 
                         weights_to_try=None, save_path=None):
    """Analyze different fusion weights."""
    if weights_to_try is None:
        weights_to_try = np.arange(0.0, 1.05, 0.1)
    
    accuracies = []
    for w_sig in weights_to_try:
        w_vis = 1 - w_sig
        probs = probs_signal * w_sig + probs_vision * w_vis
        preds = probs.argmax(axis=1)
        acc = accuracy_score(y_true, preds)
        accuracies.append(acc)
    
    plt.figure(figsize=(10, 6))
    plt.plot(weights_to_try, [a * 100 for a in accuracies], 'b-o', linewidth=2)
    plt.xlabel('Signal Expert Weight')
    plt.ylabel('Accuracy (%)')
    plt.title('Dual Ensemble: Accuracy vs. Fusion Weights')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    
    # Mark best weight
    best_idx = np.argmax(accuracies)
    best_w = weights_to_try[best_idx]
    best_acc = accuracies[best_idx] * 100
    plt.axvline(best_w, color='r', linestyle='--', label=f'Best: {best_w:.1f} ({best_acc:.2f}%)')
    plt.legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"   Saved: {save_path}")
    plt.close()
    
    return dict(zip(weights_to_try, accuracies))


def save_results(config, metrics, probs_signal, probs_vision, final_probs, 
                 final_preds, y_true, output_dir):
    """Save all results to output directory."""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save predictions
    if config['output'].get('save_predictions', True):
        np.save(f"{output_dir}/predictions_{timestamp}.npy", final_preds)
    
    # Save probabilities
    if config['output'].get('save_probabilities', True):
        np.save(f"{output_dir}/probabilities_signal_{timestamp}.npy", probs_signal)
        np.save(f"{output_dir}/probabilities_vision_{timestamp}.npy", probs_vision)
        np.save(f"{output_dir}/probabilities_ensemble_{timestamp}.npy", final_probs)
    
    # Save confusion matrix
    if config['output'].get('save_confusion_matrix', True):
        np.save(f"{output_dir}/confusion_matrix_{timestamp}.npy", metrics['confusion_matrix'])
    
    # Save report
    with open(f"{output_dir}/report_{timestamp}.txt", 'w') as f:
        f.write(f"Dual Ensemble Evaluation Results\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Signal Weight: {config['ensemble']['signal_weight']}\n")
        f.write(f"Vision Weight: {config['ensemble']['vision_weight']}\n")
        f.write(f"\nAccuracy: {metrics['accuracy']*100:.4f}%\n")
        f.write(f"\nClassification Report:\n{metrics['report']}")
    
    print(f"\n   Results saved to: {output_dir}")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description='Dual Ensemble Evaluation')
    parser.add_argument('--config', type=str, default='configs/dual_ensemble.yaml',
                       help='Path to config file')
    parser.add_argument('--signal_weight', type=float, default=None,
                       help='Override signal expert weight')
    parser.add_argument('--vision_weight', type=float, default=None,
                       help='Override vision expert weight')
    parser.add_argument('--analyze_weights', action='store_true',
                       help='Analyze different fusion weights')
    parser.add_argument('--no_plot', action='store_true',
                       help='Disable plotting')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override weights if provided
    if args.signal_weight is not None:
        config['ensemble']['signal_weight'] = args.signal_weight
    if args.vision_weight is not None:
        config['ensemble']['vision_weight'] = args.vision_weight
    
    # Ensure weights sum to 1
    total = config['ensemble']['signal_weight'] + config['ensemble']['vision_weight']
    if abs(total - 1.0) > 0.01:
        print(f"Warning: Weights sum to {total:.2f}, normalizing...")
        config['ensemble']['signal_weight'] /= total
        config['ensemble']['vision_weight'] /= total
    
    # Setup device
    device = torch.device(config['inference'].get('device', 'cuda') 
                         if torch.cuda.is_available() else 'cpu')
    print(f"{'='*70}")
    print(f"DUAL EXPERT ENSEMBLE EVALUATION")
    print(f"{'='*70}")
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load data
    X_test, y_test = load_data(config)
    
    # Load models
    signal_model, vision_model = load_models(config, device)
    
    # Signal Expert inference
    batch_size_sig = config['inference']['batch_size_signal']
    probs_signal, preds_signal = run_signal_expert(signal_model, X_test, batch_size_sig, device)
    acc_signal = accuracy_score(y_test, preds_signal)
    print(f"   Signal Expert Accuracy: {acc_signal*100:.4f}%")
    
    # Vision Expert inference  
    batch_size_vis = config['inference']['batch_size_vision']
    img_size = config['scalogram']['img_size']
    probs_vision, preds_vision = run_vision_expert(vision_model, X_test, batch_size_vis, 
                                                    img_size, device)
    acc_vision = accuracy_score(y_test, preds_vision)
    print(f"   Vision Expert Accuracy: {acc_vision*100:.4f}%")
    
    # Fusion
    signal_weight = config['ensemble']['signal_weight']
    vision_weight = config['ensemble']['vision_weight']
    final_probs, final_preds = fuse_predictions(probs_signal, probs_vision, 
                                                 signal_weight, vision_weight)
    
    # Metrics
    class_names = config['data']['class_names']
    metrics = compute_metrics(y_test, final_preds, class_names)
    
    print(f"\n{'='*70}")
    print(f"DUAL ENSEMBLE ACCURACY: {metrics['accuracy']*100:.4f}%")
    print(f"{'='*70}")
    print(f"\nClassification Report:")
    print(metrics['report'])
    print(f"\nConfusion Matrix:")
    print(metrics['confusion_matrix'])
    
    # Save and plot
    output_dir = config['experiment']['output_dir']
    
    if config['output'].get('plot_results', True) and not args.no_plot:
        os.makedirs(output_dir, exist_ok=True)
        plot_confusion_matrix(metrics['confusion_matrix'], class_names,
                             f"{output_dir}/confusion_matrix.png")
    
    # Weight analysis
    if args.analyze_weights:
        print("\n[Extra] Analyzing Different Fusion Weights...")
        weight_results = plot_weight_analysis(probs_signal, probs_vision, y_test,
                                              save_path=f"{output_dir}/weight_analysis.png")
        print("\n   Weight Analysis Results:")
        for w, acc in weight_results.items():
            print(f"     Signal {w:.1f} | Vision {1-w:.1f} -> {acc*100:.4f}%")
    
    # Save results
    save_results(config, metrics, probs_signal, probs_vision, final_probs,
                final_preds, y_test, output_dir)
    
    print(f"\n{'='*70}")
    print("Evaluation Complete!")
    print(f"{'='*70}")
    
    return metrics['accuracy']


if __name__ == "__main__":
    main()
