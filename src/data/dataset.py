"""
ECG Dataset Module with Flexible Options

Loads pre-balanced data (SMOTE or ADASYN), then splits into train/val/test.
Pre-balancing ensures consistent class distribution matching original research.

Usage:
    from src.data import ECGDataModule
    
    # Load SMOTE-balanced data
    data = ECGDataModule(mode='combined', balancing_method='smote')
    
    # Use ADASYN instead of SMOTE
    data = ECGDataModule(mode='combined', balancing_method='adasyn')
    
    # Limit samples per class (e.g., 10k per class = 50k total)
    data = ECGDataModule(mode='combined', samples_per_class=10000)
    
    # Get data loaders
    loaders = data.get_signal_loaders()  # For 1D models
    loaders = data.get_scalogram_loaders()  # For 2D vision models
"""

import os
import numpy as np
from collections import Counter
from typing import Tuple, Dict, Optional, List, Union

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, StratifiedKFold
import pywt
import cv2
from torchvision import transforms


# ============================================================================
# CONFIGURATION
# ============================================================================

class DataConfig:
    """Dataset paths and configuration."""
    
    # Base directories
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_DIR = os.path.join(BASE_DIR, 'balanced_data')
    
    # SMOTE-balanced datasets
    SMOTE_PATHS = {
        'combined': {
            'X': os.path.join(DATA_DIR, 'X_balanced_rpeak_smote.npy'),
            'y': os.path.join(DATA_DIR, 'y_balanced_rpeak_smote.npy')
        },
        'mitbih': {
            'X': os.path.join(DATA_DIR, 'X_mitbih_rpeak_smote.npy'),
            'y': os.path.join(DATA_DIR, 'y_mitbih_rpeak_smote.npy')
        },
        'incart': {
            'X': os.path.join(DATA_DIR, 'X_incart_rpeak_smote.npy'),
            'y': os.path.join(DATA_DIR, 'y_incart_rpeak_smote.npy')
        }
    }
    
    # ADASYN-balanced datasets (legacy - use only if you know what you're doing)
    ADASYN_PATHS = {
        'combined': {
            'X': os.path.join(DATA_DIR, 'X_balanced_rpeak_adasyn.npy'),
            'y': os.path.join(DATA_DIR, 'y_balanced_rpeak_adasyn.npy')
        },
        'mitbih': {
            'X': os.path.join(DATA_DIR, 'X_mitbih_rpeak_adasyn.npy'),
            'y': os.path.join(DATA_DIR, 'y_mitbih_rpeak_adasyn.npy')
        },
        'incart': {
            'X': os.path.join(DATA_DIR, 'X_incart_rpeak_adasyn.npy'),
            'y': os.path.join(DATA_DIR, 'y_incart_rpeak_adasyn.npy')
        }
    }
    
    # Class information
    NUM_CLASSES = 5
    CLASS_NAMES = ['Normal (N)', 'Supraventricular (S)', 'Ventricular (V)', 
                   'Fusion (F)', 'Unknown/Paced (Q)']
    CLASS_LABELS = ['N', 'S', 'V', 'F', 'Q']
    SEGMENT_LENGTH = 360  # Samples per segment (1 second at 360Hz)


# ============================================================================
# DATASET CLASSES
# ============================================================================

class ECGDataset(Dataset):
    """PyTorch Dataset for 1D ECG signals."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray, transform=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.y)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.X[idx]
        if self.transform:
            x = self.transform(x)
        return x, self.y[idx]


class ScalogramDataset(Dataset):
    """
    Dataset that converts 1D ECG signals to 2D scalograms using CWT.
    
    Used for EfficientNet/Vision models (Paper 2).
    """
    
    def __init__(self, X: np.ndarray, y: np.ndarray = None, 
                 img_size: int = 224, wavelet: str = 'morl',
                 scales: np.ndarray = None, cache: bool = False,
                 cache_dir: str = None, split_name: str = 'train',
                 precompute: bool = True):
        self.X = X
        self.y = torch.LongTensor(y) if y is not None else None
        self.img_size = img_size
        self.wavelet = wavelet
        self.scales = scales if scales is not None else np.arange(1, 65)
        self.cache = cache
        self.cache_dir = cache_dir
        self.split_name = split_name
        self._scalograms = None
        
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Pre-compute all scalograms for speed
        if precompute:
            self._precompute_scalograms()
    
    def _precompute_scalograms(self):
        """Pre-compute all scalograms with progress bar."""
        from tqdm import tqdm
        print(f"Pre-computing {len(self.X)} scalograms ({self.split_name})...")
        
        scalograms = []
        for i in tqdm(range(len(self.X)), desc=f"Scalograms ({self.split_name})", 
                      ncols=80, leave=True):
            rgb_img = self._compute_scalogram(self.X[i])
            img_tensor = self.normalize(rgb_img)
            scalograms.append(img_tensor)
        
        self._scalograms = scalograms
        print(f"Pre-computed {len(scalograms)} scalograms.")
    
    def _compute_scalogram(self, signal: np.ndarray) -> np.ndarray:
        """Compute single scalogram from signal."""
        coeffs, _ = pywt.cwt(signal, self.scales, self.wavelet)
        scalogram = np.abs(coeffs)
        
        resized = cv2.resize(scalogram, (self.img_size, self.img_size), 
                            interpolation=cv2.INTER_CUBIC)
        
        resized = (resized - resized.min()) / (resized.max() - resized.min() + 1e-8)
        rgb_img = np.stack((resized,) * 3, axis=-1).astype(np.float32)
        return rgb_img
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int):
        # Use pre-computed scalograms if available
        if self._scalograms is not None:
            img_tensor = self._scalograms[idx]
        else:
            rgb_img = self._compute_scalogram(self.X[idx])
            img_tensor = self.normalize(rgb_img)
        
        if self.y is not None:
            return img_tensor, self.y[idx]
        return img_tensor

# ============================================================================
# DATA MODULE (Main Interface)
# ============================================================================

class ECGDataModule:
    """
    Main data module for ECG classification experiments.
    
    Handles:
    - Loading pre-balanced datasets
    - Train/val/test splitting
    - DataLoader creation for both 1D and 2D models
    
    Args:
        mode: Dataset mode ('combined', 'mitbih', 'incart', 
              'cross_mit_incart', 'cross_incart_mit')
        test_size: Fraction for test set (default: 0.20)
        val_size: Fraction of train for validation (default: 0.15)
        seed: Random seed
        balancing_method: 'smote' or 'adasyn' - which pre-balanced file to load
        samples_per_class: Samples per class (None = use all, e.g., 10000 = 50k total for 5 classes)
        verbose: Print info
    """
    
    VALID_MODES = ['combined', 'mitbih', 'incart', 'cross_mit_incart', 'cross_incart_mit']
    
    def __init__(self, 
                 mode: str = 'combined',
                 test_size: float = 0.20,
                 val_size: float = 0.15,
                 seed: int = 42,
                 balancing_method: str = 'smote',
                 samples_per_class: int = None,
                 verbose: bool = True,
                 # Legacy/ignored parameters for compatibility
                 apply_smote: bool = None,
                 smote_target: int = None,
                 balancing_target: int = None,
                 use_raw: bool = None,
                 max_samples: int = None):
        
        if mode not in self.VALID_MODES:
            raise ValueError(f"Invalid mode '{mode}'. Valid: {self.VALID_MODES}")
        
        self.mode = mode
        self.test_size = test_size
        self.val_size = val_size
        self.seed = seed
        self.balancing_method = balancing_method if balancing_method in ['smote', 'adasyn'] else 'smote'
        self.samples_per_class = samples_per_class
        self.verbose = verbose
        
        # Data splits
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None
        
        # Full dataset (for K-fold)
        self.X_full = None
        self.y_full = None
        
        self._load_data()
    
    def _log(self, msg: str):
        if self.verbose:
            print(msg)
    
    def _load_data(self):
        """Load pre-balanced data and split."""
        self._log(f"\n{'='*60}")
        self._log(f"Loading Data - Mode: {self.mode.upper()}")
        self._log(f"Balancing: {self.balancing_method.upper()} (pre-balanced)")
        self._log(f"{'='*60}")
        
        if self.mode in ['combined', 'mitbih', 'incart']:
            self._load_single_mode(self.mode)
        elif self.mode == 'cross_mit_incart':
            self._load_cross_mode('mitbih', 'incart')
        elif self.mode == 'cross_incart_mit':
            self._load_cross_mode('incart', 'mitbih')
    
    def _get_paths(self, dataset: str) -> Dict[str, str]:
        """Get paths for a dataset based on balancing_method."""
        if self.balancing_method == 'adasyn':
            return DataConfig.ADASYN_PATHS[dataset]
        else:
            return DataConfig.SMOTE_PATHS[dataset]
    
    def _sample_per_class(self, X: np.ndarray, y: np.ndarray, n_per_class: int) -> Tuple[np.ndarray, np.ndarray]:
        """Sample exactly n_per_class samples from each class."""
        rng = np.random.RandomState(self.seed)
        indices = []
        
        for cls in np.unique(y):
            cls_indices = np.where(y == cls)[0]
            n_available = len(cls_indices)
            n_sample = min(n_per_class, n_available)
            
            sampled = rng.choice(cls_indices, size=n_sample, replace=False)
            indices.extend(sampled)
        
        indices = np.array(indices)
        rng.shuffle(indices)
        
        return X[indices], y[indices]
    
    def _load_single_mode(self, dataset: str):
        """Load single dataset and split."""
        paths = self._get_paths(dataset)
        
        if not os.path.exists(paths['X']):
            raise FileNotFoundError(
                f"Pre-balanced dataset not found: {paths['X']}\n"
                f"Make sure balancing files exist in balanced_data/"
            )
        
        X = np.load(paths['X'])
        y = np.load(paths['y']).astype(int)
        
        self._log(f"Loaded: {len(y)} samples")
        
        # Limit samples per class if requested
        if self.samples_per_class:
            X, y = self._sample_per_class(X, y, self.samples_per_class)
            actual_per_class = {cls: int((y == cls).sum()) for cls in np.unique(y)}
            self._log(f"Sampled to: {len(y)} samples (requested {self.samples_per_class}/class, actual: {min(actual_per_class.values())}/class)")
        
        self.X_full = X
        self.y_full = y
        
        self._log(f"Class distribution: {dict(Counter(y))}")
        
        # Split into train+val and test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.seed, stratify=y
        )
        
        # Split train into train and val
        actual_val_size = self.val_size / (1 - self.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=actual_val_size,
            random_state=self.seed, stratify=y_train_val
        )
        
        self.X_train, self.y_train = X_train, y_train
        self.X_val, self.y_val = X_val, y_val
        self.X_test, self.y_test = X_test, y_test
        
        self._print_split_info()
    
    def _load_cross_mode(self, train_dataset: str, test_dataset: str):
        """Load cross-database: train on one, test on another."""
        train_paths = self._get_paths(train_dataset)
        test_paths = self._get_paths(test_dataset)
        
        for p in [train_paths['X'], test_paths['X']]:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Dataset not found: {p}")
        
        # Load datasets
        X_train_full = np.load(train_paths['X'])
        y_train_full = np.load(train_paths['y']).astype(int)
        
        X_test = np.load(test_paths['X'])
        y_test = np.load(test_paths['y']).astype(int)
        
        self._log(f"Train from {train_dataset.upper()}: {len(y_train_full)} samples")
        self._log(f"Test from {test_dataset.upper()}: {len(y_test)} samples")
        
        # Split train into train/val
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=self.val_size,
            random_state=self.seed, stratify=y_train_full
        )
        
        self.X_train, self.y_train = X_train, y_train
        self.X_val, self.y_val = X_val, y_val
        self.X_test, self.y_test = X_test, y_test
        
        self.X_full = np.concatenate([X_train_full, X_test])
        self.y_full = np.concatenate([y_train_full, y_test])
        
        self._print_split_info()
    
    def _print_split_info(self):
        """Print data split summary."""
        self._log(f"\nData Split Summary:")
        self._log(f"  Train: {len(self.y_train):,} samples | {dict(Counter(self.y_train))}")
        self._log(f"  Val:   {len(self.y_val):,} samples")
        self._log(f"  Test:  {len(self.y_test):,} samples")
    
    # =========================================================================
    # PUBLIC INTERFACE
    # =========================================================================
    
    def get_signal_loaders(self, batch_size: int = 2048, 
                           num_workers: int = 4) -> Dict[str, DataLoader]:
        """
        Get DataLoaders for 1D signal models (InceptionTime, etc.).
        
        Returns:
            Dict with 'train', 'val', 'test' DataLoaders
        """
        train_ds = ECGDataset(self.X_train, self.y_train)
        val_ds = ECGDataset(self.X_val, self.y_val)
        test_ds = ECGDataset(self.X_test, self.y_test)
        
        return {
            'train': DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=True,
                               persistent_workers=num_workers > 0),
            'val': DataLoader(val_ds, batch_size=batch_size*2, shuffle=False,
                             num_workers=num_workers, pin_memory=True),
            'test': DataLoader(test_ds, batch_size=batch_size*2, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
        }
    
    def get_scalogram_loaders(self, batch_size: int = 256,
                              num_workers: int = 4,
                              img_size: int = 224) -> Dict[str, DataLoader]:
        """
        Get DataLoaders for 2D scalogram models (EfficientNet, etc.).
        
        Returns:
            Dict with 'train', 'val', 'test' DataLoaders
        """
        self._log("\nGenerating scalograms on-the-fly...")
        train_ds = ScalogramDataset(self.X_train, self.y_train, img_size=img_size)
        val_ds = ScalogramDataset(self.X_val, self.y_val, img_size=img_size)
        test_ds = ScalogramDataset(self.X_test, self.y_test, img_size=img_size)
        
        return {
            'train': DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=True),
            'val': DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True),
            'test': DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
        }
    
    def get_splits(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Get raw numpy arrays for custom use."""
        return {
            'train': (self.X_train, self.y_train),
            'val': (self.X_val, self.y_val),
            'test': (self.X_test, self.y_test)
        }
    
    def get_kfold_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get full dataset for K-fold cross-validation.
        
        Note: Data is already pre-balanced. For proper K-fold validation,
        consider using raw data with per-fold balancing.
        """
        return self.X_full, self.y_full
    
    @property
    def num_classes(self) -> int:
        return DataConfig.NUM_CLASSES
    
    @property
    def class_names(self) -> List[str]:
        return DataConfig.CLASS_NAMES
    
    @property
    def segment_length(self) -> int:
        return DataConfig.SEGMENT_LENGTH


# ============================================================================
# K-FOLD HELPER
# ============================================================================

def apply_balancing_to_fold(X_train: np.ndarray, y_train: np.ndarray, 
                            seed: int = 42, target: int = None,
                            method: str = 'smote') -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply SMOTE or ADASYN to a single fold's training data.
    
    Use this inside your K-fold loop to prevent data leakage.
    
    Args:
        method: 'smote' or 'adasyn'
    """
    from imblearn.over_sampling import SMOTE, ADASYN
    from imblearn.under_sampling import RandomUnderSampler
    
    class_counts = Counter(y_train)
    
    if target is None:
        target = int(np.mean(list(class_counts.values())))
        target = max(target, max(class_counts.values()))
    
    under_strategy = {c: min(v, target) for c, v in class_counts.items()}
    rus = RandomUnderSampler(sampling_strategy=under_strategy, random_state=seed)
    X_under, y_under = rus.fit_resample(X_train, y_train)
    
    oversample_strategy = {c: target for c in np.unique(y_under)}
    
    if method == 'adasyn':
        try:
            oversampler = ADASYN(sampling_strategy=oversample_strategy, random_state=seed)
            X_balanced, y_balanced = oversampler.fit_resample(X_under, y_under)
        except ValueError:
            # Fallback to SMOTE if ADASYN fails
            oversampler = SMOTE(sampling_strategy=oversample_strategy, random_state=seed)
            X_balanced, y_balanced = oversampler.fit_resample(X_under, y_under)
    else:
        oversampler = SMOTE(sampling_strategy=oversample_strategy, random_state=seed)
        X_balanced, y_balanced = oversampler.fit_resample(X_under, y_under)
    
    perm = np.random.RandomState(seed).permutation(len(y_balanced))
    return X_balanced[perm], y_balanced[perm]


# Legacy alias
def apply_smote_to_fold(X_train: np.ndarray, y_train: np.ndarray, 
                        seed: int = 42, target: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """Legacy function - use apply_balancing_to_fold instead."""
    return apply_balancing_to_fold(X_train, y_train, seed, target, method='smote')


if __name__ == "__main__":
    # Quick test
    data = ECGDataModule(mode='combined', balancing_method='smote', verbose=True)
    loaders = data.get_signal_loaders(batch_size=512)
    
    for batch_x, batch_y in loaders['train']:
        print(f"Batch shape: {batch_x.shape}, Labels shape: {batch_y.shape}")
        break
