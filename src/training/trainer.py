"""
Training Module for ECG Classification

Features:
- Standard training with early stopping
- K-Fold cross-validation (with proper SMOTE handling)
- Mixed precision training
- Learning rate scheduling
- Checkpointing
- Experiment logging

Usage:
    from src.training import Trainer, KFoldTrainer
    
    # Standard training
    trainer = Trainer(model, train_loader, val_loader)
    trainer.train(epochs=100)
    
    # K-Fold CV
    kfold_trainer = KFoldTrainer(model_cls, data_module, n_splits=10)
    results = kfold_trainer.run()
"""

import os
import time
import copy
import json
from pathlib import Path
from typing import Dict, List, Optional, Callable, Tuple, Any, Type
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import autocast, GradScaler
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm


# ============================================================================
# TRAINING UTILITIES
# ============================================================================

class EarlyStopping:
    """
    Early stopping with learning rate reduction before stopping.
    
    Behavior:
    1. When no improvement for `lr_patience` epochs -> reduce LR by `factor`
    2. After `max_lr_reductions` with no improvement -> stop training
    
    This is more effective than simple early stopping as it gives the model
    multiple chances to escape local minima with lower learning rates.
    """
    
    def __init__(self, 
                 patience: int = 10,
                 lr_patience: int = 5,
                 min_delta: float = 0.0, 
                 mode: str = 'min', 
                 checkpoint_path: str = None,
                 factor: float = 0.5,
                 min_lr: float = 1e-7,
                 max_lr_reductions: int = 3):
        """
        Args:
            patience: Total patience before stopping (after all LR reductions)
            lr_patience: Epochs to wait before reducing LR
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for accuracy
            checkpoint_path: Path to save best model
            factor: Factor to reduce LR by (new_lr = lr * factor)
            min_lr: Minimum learning rate threshold
            max_lr_reductions: Maximum number of LR reductions before stopping
        """
        self.patience = patience
        self.lr_patience = lr_patience
        self.min_delta = min_delta
        self.mode = mode
        self.checkpoint_path = checkpoint_path
        self.factor = factor
        self.min_lr = min_lr
        self.max_lr_reductions = max_lr_reductions
        
        self.best = float('inf') if mode == 'min' else float('-inf')
        self.counter = 0  # Epochs since improvement
        self.lr_reduction_count = 0
        self.best_epoch = 0
        self.best_state = None
        self.stopped = False
        self.lr_reduced_this_cycle = False
        self._optimizer = None
    
    def set_optimizer(self, optimizer):
        """Set optimizer for LR reduction."""
        self._optimizer = optimizer
    
    def _is_improvement(self, value: float) -> bool:
        """Check if value is an improvement."""
        if self.mode == 'min':
            return value < self.best - self.min_delta
        else:
            return value > self.best + self.min_delta
    
    def _reduce_lr(self) -> bool:
        """Reduce learning rate. Returns True if LR was reduced."""
        if self._optimizer is None:
            return False
        
        reduced = False
        for param_group in self._optimizer.param_groups:
            old_lr = param_group['lr']
            new_lr = max(old_lr * self.factor, self.min_lr)
            if new_lr < old_lr:
                param_group['lr'] = new_lr
                reduced = True
        
        if reduced:
            self.lr_reduction_count += 1
            current_lr = self._optimizer.param_groups[0]['lr']
            print(f"    -> Reducing LR to {current_lr:.2e} (reduction {self.lr_reduction_count}/{self.max_lr_reductions})")
        
        return reduced
    
    def get_current_lr(self) -> float:
        """Get current learning rate."""
        if self._optimizer is not None:
            return self._optimizer.param_groups[0]['lr']
        return 0.0
    
    def __call__(self, value: float, epoch: int, model: nn.Module) -> bool:
        """
        Check for early stopping.
        
        Returns:
            True if training should stop
        """
        if self._is_improvement(value):
            self.best = value
            self.counter = 0
            self.best_epoch = epoch
            self.best_state = copy.deepcopy(model.state_dict())
            self.lr_reduced_this_cycle = False
            
            if self.checkpoint_path:
                torch.save(model.state_dict(), self.checkpoint_path)
        else:
            self.counter += 1
            
            # Check if we should reduce LR
            if self.counter >= self.lr_patience and not self.lr_reduced_this_cycle:
                current_lr = self.get_current_lr()
                
                if current_lr > self.min_lr and self.lr_reduction_count < self.max_lr_reductions:
                    self._reduce_lr()
                    self.lr_reduced_this_cycle = True
                    self.counter = 0  # Reset counter after LR reduction
        
        # Check if we should stop
        if self.counter >= self.patience:
            # Only stop if we've exhausted LR reductions
            if self.lr_reduction_count >= self.max_lr_reductions or self.get_current_lr() <= self.min_lr:
                self.stopped = True
                return True
            else:
                # Try another LR reduction
                if self._reduce_lr():
                    self.counter = 0
                    return False
                else:
                    self.stopped = True
                    return True
        
        return False
    
    def load_best(self, model: nn.Module):
        """Load best model state."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


class WarmupCosineScheduler:
    """Learning rate scheduler with warmup and cosine annealing."""
    
    def __init__(self, optimizer: optim.Optimizer, warmup_epochs: int,
                 total_epochs: int, min_lr: float = 1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
    
    def step(self, epoch: int):
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * (1 + np.cos(np.pi * progress)) / 2
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr


def get_class_weights(y: np.ndarray, device: torch.device) -> torch.Tensor:
    """Compute inverse frequency class weights."""
    from collections import Counter
    
    counts = Counter(y)
    total = len(y)
    n_classes = len(counts)
    
    weights = torch.zeros(n_classes, device=device)
    for c in range(n_classes):
        weights[c] = total / (n_classes * counts.get(c, 1))
    
    return weights


# ============================================================================
# MAIN TRAINER
# ============================================================================

class Trainer:
    """
    Main trainer for ECG classification models.
    
    Features:
    - Mixed precision training (AMP)
    - Early stopping with checkpointing
    - Learning rate scheduling
    - Class-weighted loss
    - Progress tracking
    
    Args:
        model: PyTorch model
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        device: torch.device
        optimizer: Optimizer (default: AdamW)
        lr: Learning rate
        weight_decay: L2 regularization
        use_amp: Use automatic mixed precision
        use_class_weights: Weight loss by class frequency
        checkpoint_dir: Directory for checkpoints
    """
    
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 device: torch.device = None,
                 optimizer: optim.Optimizer = None,
                 lr: float = 1e-3,
                 weight_decay: float = 1e-4,
                 use_amp: bool = True,
                 use_class_weights: bool = True,
                 checkpoint_dir: str = 'checkpoints'):
        
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.use_amp = use_amp and torch.cuda.is_available()
        
        # Optimizer
        self.optimizer = optimizer or optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        
        # Loss
        if use_class_weights:
            # Get labels efficiently from dataset (avoid iterating through loader)
            if hasattr(train_loader.dataset, 'y') and train_loader.dataset.y is not None:
                # ScalogramDataset or custom dataset with y attribute
                labels = train_loader.dataset.y
                all_labels = labels.numpy() if isinstance(labels, torch.Tensor) else np.array(labels)
            elif hasattr(train_loader.dataset, 'tensors'):
                # TensorDataset
                all_labels = train_loader.dataset.tensors[1].numpy()
            else:
                # Fallback: iterate through loader (slow for large datasets)
                print("  Computing class weights (this may take a moment)...")
                all_labels = []
                for _, labels in train_loader:
                    all_labels.extend(labels.numpy())
                all_labels = np.array(all_labels)
            
            weights = get_class_weights(all_labels, self.device)
            self.criterion = nn.CrossEntropyLoss(weight=weights)
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # AMP scaler
        self.scaler = GradScaler('cuda') if self.use_amp else None
        
        # Checkpointing
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # History
        self.history = defaultdict(list)
    
    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_x, batch_y in self.train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with autocast('cuda'):
                    outputs = self.model(batch_x)
                    loss = self.criterion(outputs, batch_y)
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
            
            total_loss += loss.item() * batch_x.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(batch_y).sum().item()
            total += batch_x.size(0)
        
        return total_loss / total, correct / total
    
    @torch.no_grad()
    def validate(self) -> Tuple[float, float]:
        """Validate model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_x, batch_y in self.val_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            if self.use_amp:
                with autocast('cuda'):
                    outputs = self.model(batch_x)
                    loss = self.criterion(outputs, batch_y)
            else:
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
            
            total_loss += loss.item() * batch_x.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(batch_y).sum().item()
            total += batch_x.size(0)
        
        return total_loss / total, correct / total
    
    def train(self,
              epochs: int = 100,
              patience: int = 15,
              lr_patience: int = 5,
              warmup_epochs: int = 5,
              lr_reduce_factor: float = 0.5,
              min_lr: float = 1e-7,
              max_lr_reductions: int = 3,
              use_warmup_scheduler: bool = True,
              verbose: bool = True) -> Dict[str, List[float]]:
        """
        Full training loop with LR reduction before early stopping.
        
        Args:
            epochs: Maximum epochs
            patience: Epochs to wait after all LR reductions before stopping
            lr_patience: Epochs to wait before reducing LR
            warmup_epochs: Warmup epochs for LR scheduler
            verbose: Print progress
            
        Returns:
            Training history
        """
        checkpoint_path = self.checkpoint_dir / 'best_model.pt'
        early_stopping = EarlyStopping(
            patience=patience,
            lr_patience=lr_patience,
            mode='min',
            checkpoint_path=str(checkpoint_path),
            factor=lr_reduce_factor,
            min_lr=min_lr,
            max_lr_reductions=max_lr_reductions
        )
        
        # Connect optimizer for LR reduction
        early_stopping.set_optimizer(self.optimizer)
        
        scheduler = None
        if use_warmup_scheduler:
            scheduler = WarmupCosineScheduler(
                self.optimizer, warmup_epochs, epochs
            )
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Training on {self.device}")
            print(f"Epochs: {epochs} | Patience: {patience} | LR Patience: {lr_patience}")
            print(f"Max LR Reductions: {max_lr_reductions} | LR Factor: {lr_reduce_factor}")
            print(f"AMP: {self.use_amp}")
            print(f"{'='*60}\n")
        
        start_time = time.time()
        
        for epoch in range(epochs):
            # Update LR (only during warmup if using scheduler)
            if scheduler and epoch < warmup_epochs:
                lr = scheduler.step(epoch)
            else:
                lr = self.optimizer.param_groups[0]['lr']
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.validate()
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(lr)
            
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} | "
                      f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                      f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
                      f"LR: {lr:.2e}")
            
            # Early stopping with LR reduction
            if early_stopping(val_loss, epoch, self.model):
                if verbose:
                    print(f"\nEarly stopping at epoch {epoch+1} (after {early_stopping.lr_reduction_count} LR reductions)")
                break
        
        # Load best model
        early_stopping.load_best(self.model)
        
        elapsed = time.time() - start_time
        if verbose:
            print(f"\nTraining complete in {elapsed/60:.1f} minutes")
            print(f"Best epoch: {early_stopping.best_epoch+1} | Best val loss: {early_stopping.best:.4f}")
            print(f"Total LR reductions: {early_stopping.lr_reduction_count}")
        
        # Store training metadata in history
        self.history['training_time_seconds'] = elapsed
        self.history['best_epoch'] = early_stopping.best_epoch + 1
        self.history['best_val_loss'] = early_stopping.best
        self.history['lr_reductions'] = early_stopping.lr_reduction_count
        
        return dict(self.history)
    
    def save_checkpoint(self, path: str = None):
        """Save model and training state."""
        path = path or self.checkpoint_dir / 'checkpoint.pt'
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': dict(self.history),
        }, path)
    
    def load_checkpoint(self, path: str):
        """Load model and training state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = defaultdict(list, checkpoint.get('history', {}))


# ============================================================================
# K-FOLD TRAINER
# ============================================================================

class KFoldTrainer:
    """
    K-Fold Cross-Validation Trainer with flexible balancing options.
    
    IMPORTANT: Balancing is applied INSIDE each fold to prevent data leakage.
    
    Args:
        model_fn: Function that returns a new model instance
        X: Full dataset features
        y: Full dataset labels
        n_splits: Number of folds
        balancing_method: 'smote', 'adasyn', or 'none' (default: 'smote')
        balancing_target: Target samples per class (None = auto)
        apply_smote: [DEPRECATED] Use balancing_method instead
        smote_target: [DEPRECATED] Use balancing_target instead
        batch_size: Batch size
        epochs: Max epochs per fold
        patience: Early stopping patience
        lr_patience: Epochs before LR reduction
        lr_reduce_factor: LR reduction factor
        min_lr: Minimum learning rate
        max_lr_reductions: Max LR reductions
        device: torch.device
        experiment_name: Name for saving results
    """
    
    def __init__(self,
                 model_fn: Callable[[], nn.Module],
                 X: np.ndarray,
                 y: np.ndarray,
                 n_splits: int = 10,
                 balancing_method: str = 'smote',
                 balancing_target: int = None,
                 apply_smote: bool = None,  # Deprecated
                 smote_target: int = None,  # Deprecated
                 batch_size: int = 2048,
                 epochs: int = 100,
                 patience: int = 10,
                 lr_patience: int = 5,
                 lr_reduce_factor: float = 0.5,
                 min_lr: float = 1e-7,
                 max_lr_reductions: int = 3,
                 device: torch.device = None,
                 experiment_name: str = 'kfold_experiment',
                 use_scalograms: bool = False,
                 scalogram_img_size: int = 224):
        
        self.model_fn = model_fn
        self.X = X
        self.y = y
        self.n_splits = n_splits
        
        # Handle deprecated parameters
        if apply_smote is not None:
            import warnings
            warnings.warn("apply_smote is deprecated, use balancing_method instead", DeprecationWarning)
            balancing_method = 'smote' if apply_smote else 'none'
        if smote_target is not None:
            import warnings
            warnings.warn("smote_target is deprecated, use balancing_target instead", DeprecationWarning)
            balancing_target = smote_target
        
        self.balancing_method = balancing_method
        self.balancing_target = balancing_target
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.lr_patience = lr_patience
        self.lr_reduce_factor = lr_reduce_factor
        self.min_lr = min_lr
        self.max_lr_reductions = max_lr_reductions
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.experiment_name = experiment_name
        self.use_scalograms = use_scalograms
        self.scalogram_img_size = scalogram_img_size
        
        # Results storage
        self.fold_results = []
        self.fold_models = []
    
    def _apply_balancing_fold(self, X_train: np.ndarray, y_train: np.ndarray, 
                              seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """Apply SMOTE or ADASYN to a single fold's training data."""
        from imblearn.over_sampling import SMOTE, ADASYN
        from imblearn.under_sampling import RandomUnderSampler
        from collections import Counter
        
        class_counts = Counter(y_train)
        
        # Use target if specified, otherwise auto-balance
        if self.balancing_target is not None:
            target = self.balancing_target
        else:
            target = int(np.mean(list(class_counts.values())))
            target = max(target, max(class_counts.values()))
        
        # Undersample majority classes
        under_strategy = {c: min(v, target) for c, v in class_counts.items()}
        rus = RandomUnderSampler(sampling_strategy=under_strategy, random_state=seed)
        X_under, y_under = rus.fit_resample(X_train, y_train)
        
        # Oversample minority classes
        oversample_strategy = {c: target for c in np.unique(y_under)}
        
        if self.balancing_method == 'adasyn':
            try:
                oversampler = ADASYN(sampling_strategy=oversample_strategy, random_state=seed)
                X_balanced, y_balanced = oversampler.fit_resample(X_under, y_under)
            except ValueError:
                # Fallback to SMOTE if ADASYN fails
                oversampler = SMOTE(sampling_strategy=oversample_strategy, random_state=seed)
                X_balanced, y_balanced = oversampler.fit_resample(X_under, y_under)
        else:  # smote
            oversampler = SMOTE(sampling_strategy=oversample_strategy, random_state=seed)
            X_balanced, y_balanced = oversampler.fit_resample(X_under, y_under)
        
        # Shuffle
        perm = np.random.RandomState(seed).permutation(len(y_balanced))
        return X_balanced[perm], y_balanced[perm]
    
    def run(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Run K-Fold cross-validation.
        
        Returns:
            Dictionary with aggregated results
        """
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"K-Fold Cross-Validation ({self.n_splits} folds)")
            print(f"Samples: {len(self.y)} | Balancing: {self.balancing_method.upper()}")
            print(f"{'='*60}")
        
        all_preds = np.zeros_like(self.y)
        all_probs = np.zeros((len(self.y), len(np.unique(self.y))))
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(self.X, self.y)):
            if verbose:
                print(f"\n--- Fold {fold+1}/{self.n_splits} ---")
            
            # Split data
            X_train, X_val = self.X[train_idx], self.X[val_idx]
            y_train, y_val = self.y[train_idx], self.y[val_idx]
            
            # Apply balancing to training fold
            if self.balancing_method != 'none':
                X_train, y_train = self._apply_balancing_fold(X_train, y_train, seed=42+fold)
                if verbose:
                    print(f"After {self.balancing_method.upper()}: {len(y_train)} train samples")
            
            # Create DataLoaders
            if self.use_scalograms:
                # Use ScalogramDataset for 2D vision models
                from src.data.dataset import ScalogramDataset
                cache_dir = f'processed_data/scalograms/kfold_fold{fold+1}'
                train_ds = ScalogramDataset(X_train, y_train, img_size=self.scalogram_img_size,
                                           cache=True, cache_dir=cache_dir, split_name='train')
                val_ds = ScalogramDataset(X_val, y_val, img_size=self.scalogram_img_size,
                                         cache=True, cache_dir=cache_dir, split_name='val')
            else:
                # Standard 1D signal datasets
                train_ds = TensorDataset(
                    torch.FloatTensor(X_train),
                    torch.LongTensor(y_train)
                )
                val_ds = TensorDataset(
                    torch.FloatTensor(X_val),
                    torch.LongTensor(y_val)
                )
            
            train_loader = DataLoader(train_ds, batch_size=self.batch_size, 
                                      shuffle=True, num_workers=0, pin_memory=True)
            val_loader = DataLoader(val_ds, batch_size=self.batch_size*2,
                                   shuffle=False, num_workers=0, pin_memory=True)
            
            # Create fresh model
            model = self.model_fn().to(self.device)
            
            # Train
            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=self.device,
                use_class_weights=True,
                checkpoint_dir=f'checkpoints/{self.experiment_name}/fold_{fold+1}'
            )
            
            history = trainer.train(
                epochs=self.epochs,
                patience=self.patience,
                lr_patience=self.lr_patience,
                lr_reduce_factor=self.lr_reduce_factor,
                min_lr=self.min_lr,
                max_lr_reductions=self.max_lr_reductions,
                verbose=verbose
            )
            
            # Get predictions using val_loader (handles both 1D and scalogram data)
            model.eval()
            all_fold_preds = []
            all_fold_probs = []
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(self.device)
                    outputs = model(batch_x)
                    probs = torch.softmax(outputs, dim=1).cpu().numpy()
                    preds = outputs.argmax(dim=1).cpu().numpy()
                    all_fold_preds.extend(preds)
                    all_fold_probs.extend(probs)
            
            preds = np.array(all_fold_preds)
            probs = np.array(all_fold_probs)
            
            all_preds[val_idx] = preds
            all_probs[val_idx] = probs
            
            # Store results
            from sklearn.metrics import accuracy_score, f1_score
            fold_acc = accuracy_score(y_val, preds)
            fold_f1 = f1_score(y_val, preds, average='weighted')
            
            self.fold_results.append({
                'fold': fold + 1,
                'accuracy': fold_acc,
                'f1_weighted': fold_f1,
                'history': history
            })
            self.fold_models.append(model.state_dict())
            
            if verbose:
                print(f"Fold {fold+1} - Acc: {fold_acc:.4f} | F1: {fold_f1:.4f}")
        
        # Aggregate results
        from sklearn.metrics import classification_report, confusion_matrix
        
        results = {
            'n_splits': self.n_splits,
            'fold_results': self.fold_results,
            'mean_accuracy': np.mean([r['accuracy'] for r in self.fold_results]),
            'std_accuracy': np.std([r['accuracy'] for r in self.fold_results]),
            'mean_f1': np.mean([r['f1_weighted'] for r in self.fold_results]),
            'std_f1': np.std([r['f1_weighted'] for r in self.fold_results]),
            'all_predictions': all_preds,
            'all_probabilities': all_probs,
            'classification_report': classification_report(self.y, all_preds),
            'confusion_matrix': confusion_matrix(self.y, all_preds),
        }
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"K-Fold Results Summary")
            print(f"{'='*60}")
            print(f"Accuracy: {results['mean_accuracy']:.4f} ± {results['std_accuracy']:.4f}")
            print(f"F1 Score: {results['mean_f1']:.4f} ± {results['std_f1']:.4f}")
            print(f"\n{results['classification_report']}")
        
        return results
    
    def save_results(self, path: str = None):
        """Save all fold models and results."""
        path = Path(path or f'experiments/{self.experiment_name}')
        path.mkdir(parents=True, exist_ok=True)
        
        # Save fold models
        for i, state_dict in enumerate(self.fold_models):
            torch.save(state_dict, path / f'fold_{i+1}_model.pt')
        
        # Save results (excluding non-serializable)
        results_to_save = {
            'fold_results': [
                {k: v for k, v in r.items() if k != 'history'} 
                for r in self.fold_results
            ],
            'mean_accuracy': float(np.mean([r['accuracy'] for r in self.fold_results])),
            'std_accuracy': float(np.std([r['accuracy'] for r in self.fold_results])),
            'mean_f1': float(np.mean([r['f1_weighted'] for r in self.fold_results])),
            'std_f1': float(np.std([r['f1_weighted'] for r in self.fold_results])),
        }
        
        with open(path / 'results.json', 'w') as f:
            json.dump(results_to_save, f, indent=2)


# ============================================================================
# TRAINING __init__.py EXPORTS
# ============================================================================

__all__ = [
    'Trainer',
    'KFoldTrainer',
    'EarlyStopping',
    'WarmupCosineScheduler',
    'get_class_weights',
]
