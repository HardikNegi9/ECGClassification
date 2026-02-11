"""
Evaluation Metrics Module

Comprehensive metrics for ECG classification research papers:
- Per-class metrics (Precision, Recall, F1, Specificity)
- Overall metrics (Accuracy, Macro/Weighted F1, AUC-ROC)
- Paper-ready tables and visualizations
- Inference time measurement
- Statistical significance testing

Usage:
    from src.evaluation import PaperMetrics, evaluate_model
    
    metrics = PaperMetrics(y_true, y_pred, y_probs)
    metrics.print_full_report()
    metrics.export_latex_table('results.tex')
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    cohen_kappa_score,
    matthews_corrcoef,
)


# ============================================================================
# PAPER METRICS
# ============================================================================

class PaperMetrics:
    """
    Comprehensive metrics for research paper publication.
    
    Computes all standard metrics used in ECG classification papers:
    - Per-class: Precision (PPV), Recall (Sensitivity), Specificity, F1
    - Overall: Accuracy, Macro F1, Weighted F1, AUC-ROC
    - Inter-rater: Cohen's Kappa, Matthews Correlation Coefficient
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_probs: Prediction probabilities (optional, for AUC)
        class_names: Class names for display
    """
    
    DEFAULT_CLASSES = ['N', 'S', 'V', 'F', 'Q']
    
    def __init__(self,
                 y_true: np.ndarray,
                 y_pred: np.ndarray,
                 y_probs: np.ndarray = None,
                 class_names: List[str] = None):
        
        self.y_true = np.asarray(y_true)
        self.y_pred = np.asarray(y_pred)
        self.y_probs = np.asarray(y_probs) if y_probs is not None else None
        self.class_names = class_names or self.DEFAULT_CLASSES
        self.n_classes = len(self.class_names)
        
        # Compute metrics
        self._compute_metrics()
    
    def _compute_metrics(self):
        """Compute all metrics."""
        # Confusion matrix
        self.cm = confusion_matrix(self.y_true, self.y_pred)
        
        # Per-class metrics
        self.per_class = {}
        for i, name in enumerate(self.class_names):
            # True/False Positives/Negatives
            tp = self.cm[i, i]
            fp = self.cm[:, i].sum() - tp
            fn = self.cm[i, :].sum() - tp
            tn = self.cm.sum() - tp - fp - fn
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            self.per_class[name] = {
                'precision': precision,
                'recall': recall,  # Same as sensitivity
                'sensitivity': recall,
                'specificity': specificity,
                'f1': f1,
                'support': int(self.cm[i, :].sum()),
                'tp': int(tp),
                'fp': int(fp),
                'fn': int(fn),
                'tn': int(tn),
            }
        
        # Overall metrics
        self.overall = {
            'accuracy': accuracy_score(self.y_true, self.y_pred),
            'precision_macro': precision_score(self.y_true, self.y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(self.y_true, self.y_pred, average='macro', zero_division=0),
            'f1_macro': f1_score(self.y_true, self.y_pred, average='macro', zero_division=0),
            'f1_weighted': f1_score(self.y_true, self.y_pred, average='weighted', zero_division=0),
            'cohen_kappa': cohen_kappa_score(self.y_true, self.y_pred),
            'mcc': matthews_corrcoef(self.y_true, self.y_pred),
        }
        
        # AUC-ROC (if probabilities available)
        if self.y_probs is not None:
            try:
                self.overall['auc_macro'] = roc_auc_score(
                    self.y_true, self.y_probs, average='macro', multi_class='ovr'
                )
                self.overall['auc_weighted'] = roc_auc_score(
                    self.y_true, self.y_probs, average='weighted', multi_class='ovr'
                )
            except ValueError:
                self.overall['auc_macro'] = None
                self.overall['auc_weighted'] = None
    
    def get_per_class_df(self) -> pd.DataFrame:
        """Get per-class metrics as DataFrame."""
        df = pd.DataFrame(self.per_class).T
        df = df[['precision', 'recall', 'specificity', 'f1', 'support']]
        df.columns = ['Precision (PPV)', 'Sensitivity', 'Specificity', 'F1-Score', 'Support']
        return df
    
    def get_summary_dict(self) -> Dict[str, float]:
        """Get summary metrics as dict."""
        return {
            'Accuracy': self.overall['accuracy'],
            'F1 (Macro)': self.overall['f1_macro'],
            'F1 (Weighted)': self.overall['f1_weighted'],
            'AUC-ROC': self.overall.get('auc_macro'),
            "Cohen's Kappa": self.overall['cohen_kappa'],
            'MCC': self.overall['mcc'],
        }
    
    def print_full_report(self):
        """Print comprehensive report."""
        print("\n" + "="*70)
        print("CLASSIFICATION REPORT")
        print("="*70)
        
        # Per-class
        print("\nPer-Class Metrics:")
        print("-"*70)
        df = self.get_per_class_df()
        print(df.to_string(float_format=lambda x: f'{x:.4f}'))
        
        # Overall
        print("\n" + "-"*70)
        print("Overall Metrics:")
        print("-"*70)
        for name, value in self.get_summary_dict().items():
            if value is not None:
                print(f"  {name:<20}: {value:.4f}")
        
        # Confusion Matrix
        print("\n" + "-"*70)
        print("Confusion Matrix:")
        print("-"*70)
        print(f"{'':10}", end='')
        for name in self.class_names:
            print(f"{name:>8}", end='')
        print()
        for i, name in enumerate(self.class_names):
            print(f"{name:<10}", end='')
            for j in range(self.n_classes):
                print(f"{self.cm[i,j]:8d}", end='')
            print()
        
        print("="*70 + "\n")
    
    def export_latex_table(self, filepath: str = None) -> str:
        """
        Export per-class metrics as LaTeX table.
        
        Returns:
            LaTeX table string
        """
        df = self.get_per_class_df()
        
        latex = "\\begin{table}[h]\n\\centering\n"
        latex += "\\caption{Per-class Classification Performance}\n"
        latex += "\\label{tab:results}\n"
        latex += "\\begin{tabular}{lcccc}\n"
        latex += "\\hline\n"
        latex += "Class & Precision & Sensitivity & Specificity & F1-Score \\\\\n"
        latex += "\\hline\n"
        
        for idx in df.index:
            row = df.loc[idx]
            latex += f"{idx} & {row['Precision (PPV)']:.4f} & {row['Sensitivity']:.4f} & "
            latex += f"{row['Specificity']:.4f} & {row['F1-Score']:.4f} \\\\\n"
        
        latex += "\\hline\n"
        latex += f"Average & {df['Precision (PPV)'].mean():.4f} & {df['Sensitivity'].mean():.4f} & "
        latex += f"{df['Specificity'].mean():.4f} & {df['F1-Score'].mean():.4f} \\\\\n"
        latex += "\\hline\n"
        latex += "\\end{tabular}\n"
        latex += "\\end{table}"
        
        if filepath:
            with open(filepath, 'w') as f:
                f.write(latex)
        
        return latex
    
    def export_csv(self, filepath: str):
        """Export metrics to CSV."""
        df = self.get_per_class_df()
        df.to_csv(filepath)


# ============================================================================
# VISUALIZATION
# ============================================================================

class MetricsVisualizer:
    """
    Visualization tools for classification metrics.
    """
    
    def __init__(self, metrics: PaperMetrics):
        self.metrics = metrics
    
    def plot_confusion_matrix(self, figsize: Tuple[int, int] = (10, 8),
                               normalize: bool = False,
                               cmap: str = 'Blues',
                               save_path: str = None) -> plt.Figure:
        """Plot confusion matrix heatmap."""
        fig, ax = plt.subplots(figsize=figsize)
        
        cm = self.metrics.cm
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
            fmt = '.2%'
        else:
            fmt = 'd'
        
        sns.heatmap(cm, annot=True, fmt=fmt, cmap=cmap,
                   xticklabels=self.metrics.class_names,
                   yticklabels=self.metrics.class_names,
                   ax=ax)
        
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('True', fontsize=12)
        ax.set_title('Confusion Matrix', fontsize=14)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_per_class_metrics(self, figsize: Tuple[int, int] = (12, 6),
                                save_path: str = None) -> plt.Figure:
        """Plot per-class metrics as grouped bar chart."""
        fig, ax = plt.subplots(figsize=figsize)
        
        metrics = ['precision', 'recall', 'specificity', 'f1']
        x = np.arange(len(self.metrics.class_names))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            values = [self.metrics.per_class[c][metric] for c in self.metrics.class_names]
            ax.bar(x + i*width, values, width, label=metric.capitalize())
        
        ax.set_xlabel('Class')
        ax.set_ylabel('Score')
        ax.set_title('Per-Class Performance Metrics')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(self.metrics.class_names)
        ax.legend()
        ax.set_ylim([0, 1.05])
        ax.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_roc_curves(self, figsize: Tuple[int, int] = (10, 8),
                        save_path: str = None) -> plt.Figure:
        """Plot ROC curves for each class."""
        if self.metrics.y_probs is None:
            raise ValueError("Probabilities required for ROC curves")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # One-hot encode true labels
        y_true_bin = np.zeros((len(self.metrics.y_true), self.metrics.n_classes))
        y_true_bin[np.arange(len(self.metrics.y_true)), self.metrics.y_true] = 1
        
        colors = plt.cm.tab10(np.linspace(0, 1, self.metrics.n_classes))
        
        for i, (name, color) in enumerate(zip(self.metrics.class_names, colors)):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], self.metrics.y_probs[:, i])
            auc = roc_auc_score(y_true_bin[:, i], self.metrics.y_probs[:, i])
            ax.plot(fpr, tpr, color=color, label=f'{name} (AUC = {auc:.3f})')
        
        ax.plot([0, 1], [0, 1], 'k--', label='Random')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves')
        ax.legend(loc='lower right')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_learning_curves(self, history: Dict[str, List[float]],
                             figsize: Tuple[int, int] = (14, 5),
                             save_path: str = None) -> plt.Figure:
        """Plot training and validation learning curves."""
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Loss
        axes[0].plot(history['train_loss'], label='Train', linewidth=2)
        axes[0].plot(history['val_loss'], label='Validation', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Loss Curves')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[1].plot(history['train_acc'], label='Train', linewidth=2)
        axes[1].plot(history['val_acc'], label='Validation', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Accuracy Curves')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Detect overfitting
        if len(history['val_loss']) > 5:
            val_loss = np.array(history['val_loss'])
            train_loss = np.array(history['train_loss'])
            best_epoch = np.argmin(val_loss)
            
            axes[0].axvline(x=best_epoch, color='red', linestyle='--', alpha=0.7, 
                           label=f'Best: {best_epoch+1}')
            
            # Check for overfitting (val increasing while train decreasing)
            if len(val_loss) > best_epoch + 5:
                end_val = val_loss[-5:].mean()
                end_train = train_loss[-5:].mean()
                gap = end_val - end_train
                
                if gap > 0.1:
                    fig.suptitle('⚠️ Overfitting Detected', fontsize=12, color='red')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


# ============================================================================
# INFERENCE TIMING
# ============================================================================

class InferenceTimer:
    """
    Measure model inference time for benchmarking.
    """
    
    def __init__(self, model: nn.Module, device: torch.device = None):
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
    
    @torch.no_grad()
    def measure_single(self, input_shape: Tuple[int, ...] = (1, 360),
                       n_runs: int = 100, warmup: int = 10) -> Dict[str, float]:
        """
        Measure single sample inference time.
        
        Returns:
            Dict with mean, std, min, max times in milliseconds
        """
        x = torch.randn(*input_shape).to(self.device)
        
        # Warmup
        for _ in range(warmup):
            _ = self.model(x)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Measure
        times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            _ = self.model(x)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)  # ms
        
        return {
            'mean_ms': np.mean(times),
            'std_ms': np.std(times),
            'min_ms': np.min(times),
            'max_ms': np.max(times),
            'throughput_per_sec': 1000 / np.mean(times),
        }
    
    @torch.no_grad()
    def measure_batch(self, batch_sizes: List[int] = [1, 16, 32, 64, 128, 256],
                      input_length: int = 360, n_runs: int = 50) -> pd.DataFrame:
        """
        Measure inference time for different batch sizes.
        
        Returns:
            DataFrame with timing results
        """
        results = []
        
        for bs in batch_sizes:
            x = torch.randn(bs, input_length).to(self.device)
            
            # Warmup
            for _ in range(10):
                _ = self.model(x)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            times = []
            for _ in range(n_runs):
                start = time.perf_counter()
                _ = self.model(x)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                times.append((time.perf_counter() - start) * 1000)
            
            results.append({
                'batch_size': bs,
                'total_ms': np.mean(times),
                'per_sample_ms': np.mean(times) / bs,
                'throughput': bs * 1000 / np.mean(times),
            })
        
        return pd.DataFrame(results)


# ============================================================================
# EVALUATION HELPERS
# ============================================================================

@torch.no_grad()
def evaluate_model(model: nn.Module, 
                   test_loader: DataLoader,
                   device: torch.device = None,
                   return_predictions: bool = True) -> Dict[str, Any]:
    """
    Evaluate a model on test data.
    
    Args:
        model: Trained model
        test_loader: Test DataLoader
        device: torch.device
        return_predictions: Include predictions in result
        
    Returns:
        Dictionary with metrics and (optionally) predictions
    """
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_probs = []
    all_labels = []
    
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.to(device)
        outputs = model(batch_x)
        probs = torch.softmax(outputs, dim=1)
        preds = outputs.argmax(dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(batch_y.numpy())
    
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_probs = np.array(all_probs)
    
    # Compute metrics
    metrics = PaperMetrics(y_true, y_pred, y_probs)
    
    result = {
        'metrics': metrics,
        'overall': metrics.overall,
        'per_class': metrics.per_class,
        'confusion_matrix': metrics.cm,
    }
    
    if return_predictions:
        result['y_true'] = y_true
        result['y_pred'] = y_pred
        result['y_probs'] = y_probs
    
    return result


def compare_models(results: Dict[str, Dict[str, Any]], 
                   save_path: str = None) -> pd.DataFrame:
    """
    Compare multiple model results.
    
    Args:
        results: Dict mapping model names to their evaluation results
        save_path: Optional path to save comparison CSV
        
    Returns:
        Comparison DataFrame
    """
    rows = []
    for name, res in results.items():
        overall = res['overall']
        rows.append({
            'Model': name,
            'Accuracy': overall['accuracy'],
            'F1 (Macro)': overall['f1_macro'],
            'F1 (Weighted)': overall['f1_weighted'],
            'AUC-ROC': overall.get('auc_macro'),
            "Cohen's Kappa": overall['cohen_kappa'],
            'MCC': overall['mcc'],
        })
    
    df = pd.DataFrame(rows)
    df = df.sort_values('F1 (Macro)', ascending=False)
    
    if save_path:
        df.to_csv(save_path, index=False)
    
    return df


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'PaperMetrics',
    'MetricsVisualizer',
    'InferenceTimer',
    'evaluate_model',
    'compare_models',
]
