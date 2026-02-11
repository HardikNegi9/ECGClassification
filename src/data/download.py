"""
Dataset Download and Creation Utilities

Supports:
- MIT-BIH Arrhythmia Database (physionet)
- INCART Database (physionet)

Usage:
    python -m src.data.download --create-raw
"""

import os
import sys
import wfdb
import numpy as np
from collections import Counter
from typing import Dict, List, Tuple, Optional
import argparse
from pathlib import Path


# ============================================================================
# CONFIGURATION
# ============================================================================

class DownloadConfig:
    """Database configuration and paths."""
    
    BASE_DIR = Path(__file__).parent.parent.parent
    DATA_DIR = BASE_DIR / 'balanced_data'
    MITBIH_DIR = BASE_DIR / 'mit_bih_data'
    INCART_DIR = BASE_DIR / 'incart_data'
    
    # PhysioNet databases
    DATABASES = {
        'mitbih': {
            'name': 'MIT-BIH Arrhythmia',
            'physionet_id': 'mitdb',
            'fs': 360,
            'local_dir': MITBIH_DIR
        },
        'incart': {
            'name': 'INCART',
            'physionet_id': 'incartdb',
            'fs': 257,
            'local_dir': INCART_DIR
        }
    }
    
    # AAMI beat type mapping
    AAMI_MAPPING = {
        # Normal (N)
        'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0, '.': 0, 'n': 0,
        # Supraventricular (S)
        'A': 1, 'a': 1, 'J': 1, 'S': 1,
        # Ventricular (V)
        'V': 2, 'E': 2,
        # Fusion (F)
        'F': 3,
        # Unknown/Paced (Q)
        '/': 4, 'f': 4, 'Q': 4, 'U': 4, '?': 4, 'P': 4, 'p': 4
    }
    
    # Segment parameters
    TARGET_FS = 360  # Resample all to this
    SEGMENT_BEFORE = 90  # Samples before R-peak (at 360Hz)
    SEGMENT_AFTER = 270  # Samples after R-peak (at 360Hz)
    SEGMENT_LENGTH = SEGMENT_BEFORE + SEGMENT_AFTER  # 360 samples
    
    NUM_CLASSES = 5
    CLASS_NAMES = ['N', 'S', 'V', 'F', 'Q']


# ============================================================================
# DATASET DOWNLOADER
# ============================================================================

class DatasetDownloader:
    """
    Download datasets from PhysioNet.
    
    Usage:
        downloader = DatasetDownloader()
        downloader.download_mitbih()
        downloader.download_incart()
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        
    def _log(self, msg: str):
        if self.verbose:
            print(msg)
    
    def download_mitbih(self, force: bool = False) -> Path:
        """Download MIT-BIH Arrhythmia Database."""
        return self._download_database('mitbih', force)
    
    def download_incart(self, force: bool = False) -> Path:
        """Download INCART Database."""
        return self._download_database('incart', force)
    
    def _download_database(self, db_name: str, force: bool = False) -> Path:
        """Download a PhysioNet database."""
        config = DownloadConfig.DATABASES[db_name]
        local_dir = config['local_dir']
        
        # Check if already downloaded
        if local_dir.exists() and not force:
            files = list(local_dir.glob('*.dat'))
            if files:
                self._log(f"{config['name']} already downloaded ({len(files)} records)")
                return local_dir
        
        self._log(f"Downloading {config['name']} from PhysioNet...")
        local_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Get record list
            records = wfdb.io.get_record_list(config['physionet_id'])
            self._log(f"Found {len(records)} records")
            
            # Download each record
            for i, rec in enumerate(records):
                self._log(f"  Downloading {rec} ({i+1}/{len(records)})")
                wfdb.dl_database(
                    config['physionet_id'],
                    str(local_dir),
                    records=[rec]
                )
            
            self._log(f"Download complete: {local_dir}")
            return local_dir
            
        except Exception as e:
            self._log(f"Download error: {e}")
            raise


# ============================================================================
# DATASET CREATOR
# ============================================================================

class DatasetCreator:
    """
    Create processed datasets from raw PhysioNet files.
    
    Creates RAW datasets (no SMOTE) for proper train/test splitting.
    
    Usage:
        creator = DatasetCreator()
        creator.create_all_raw()
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.config = DownloadConfig
        
    def _log(self, msg: str):
        if self.verbose:
            print(msg)
    
    def create_all_raw(self):
        """Create all RAW datasets (MIT-BIH, INCART, Combined)."""
        self._log("\n" + "="*60)
        self._log("Creating RAW Datasets (No SMOTE)")
        self._log("="*60)
        
        # Ensure output directory exists
        self.config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        # Create individual datasets
        X_mit, y_mit = self.create_mitbih_raw()
        X_inc, y_inc = self.create_incart_raw()
        
        # Create combined
        self.create_combined_raw(X_mit, y_mit, X_inc, y_inc)
        
        self._log("\n" + "="*60)
        self._log("All RAW datasets created!")
        self._log("="*60)
    
    def create_mitbih_raw(self) -> Tuple[np.ndarray, np.ndarray]:
        """Process MIT-BIH to RPeak segments."""
        self._log("\n--- Processing MIT-BIH ---")
        
        data_dir = self.config.MITBIH_DIR
        if not data_dir.exists():
            raise FileNotFoundError(f"MIT-BIH data not found: {data_dir}")
        
        X, y = self._process_database(data_dir, fs=360)
        
        # Save
        out_X = self.config.DATA_DIR / 'X_mitbih_raw.npy'
        out_y = self.config.DATA_DIR / 'y_mitbih_raw.npy'
        np.save(out_X, X)
        np.save(out_y, y)
        
        self._log(f"Saved: {out_X}")
        return X, y
    
    def create_incart_raw(self) -> Tuple[np.ndarray, np.ndarray]:
        """Process INCART to RPeak segments."""
        self._log("\n--- Processing INCART ---")
        
        data_dir = self.config.INCART_DIR
        if not data_dir.exists():
            raise FileNotFoundError(f"INCART data not found: {data_dir}")
        
        X, y = self._process_database(data_dir, fs=257)
        
        # Save
        out_X = self.config.DATA_DIR / 'X_incart_raw.npy'
        out_y = self.config.DATA_DIR / 'y_incart_raw.npy'
        np.save(out_X, X)
        np.save(out_y, y)
        
        self._log(f"Saved: {out_X}")
        return X, y
    
    def create_combined_raw(self, X_mit: np.ndarray = None, y_mit: np.ndarray = None,
                            X_inc: np.ndarray = None, y_inc: np.ndarray = None):
        """Combine MIT-BIH and INCART datasets."""
        self._log("\n--- Creating Combined Dataset ---")
        
        # Load if not provided
        if X_mit is None:
            X_mit = np.load(self.config.DATA_DIR / 'X_mitbih_raw.npy')
            y_mit = np.load(self.config.DATA_DIR / 'y_mitbih_raw.npy')
        if X_inc is None:
            X_inc = np.load(self.config.DATA_DIR / 'X_incart_raw.npy')
            y_inc = np.load(self.config.DATA_DIR / 'y_incart_raw.npy')
        
        # Combine
        X_combined = np.vstack([X_mit, X_inc])
        y_combined = np.concatenate([y_mit, y_inc])
        
        # Shuffle
        perm = np.random.permutation(len(y_combined))
        X_combined = X_combined[perm]
        y_combined = y_combined[perm]
        
        # Save
        out_X = self.config.DATA_DIR / 'X_combined_raw.npy'
        out_y = self.config.DATA_DIR / 'y_combined_raw.npy'
        np.save(out_X, X_combined)
        np.save(out_y, y_combined)
        
        self._log(f"Combined: {len(y_combined)} samples")
        self._log(f"Distribution: {dict(Counter(y_combined))}")
        self._log(f"Saved: {out_X}")
        
        return X_combined, y_combined
    
    def _process_database(self, data_dir: Path, fs: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process a PhysioNet database into RPeak-centered segments.
        
        Steps:
        1. Read each record and annotations
        2. Extract fixed-length segments around each R-peak
        3. Map beat labels to AAMI classes
        4. Resample if needed (for INCART 257Hz -> 360Hz)
        """
        from scipy import signal as sp_signal
        
        segments = []
        labels = []
        
        # Find records
        records = sorted(set(
            p.stem for p in data_dir.glob('*.dat')
        ))
        
        self._log(f"Found {len(records)} records in {data_dir}")
        
        for rec_name in records:
            try:
                rec_path = str(data_dir / rec_name)
                
                # Read record and annotation
                record = wfdb.rdrecord(rec_path)
                ann = wfdb.rdann(rec_path, 'atr')
                
                # Use first channel (MLII typically)
                sig = record.p_signal[:, 0]
                
                # Resample if not 360Hz
                if fs != self.config.TARGET_FS:
                    num_samples = int(len(sig) * self.config.TARGET_FS / fs)
                    sig = sp_signal.resample(sig, num_samples)
                    # Scale annotation positions
                    scale = self.config.TARGET_FS / fs
                    sample_positions = (np.array(ann.sample) * scale).astype(int)
                else:
                    sample_positions = ann.sample
                
                # Normalize signal
                sig = (sig - np.mean(sig)) / (np.std(sig) + 1e-8)
                
                # Extract segments
                before = self.config.SEGMENT_BEFORE
                after = self.config.SEGMENT_AFTER
                seg_len = self.config.SEGMENT_LENGTH
                
                for idx, samp in enumerate(sample_positions):
                    # Get beat symbol
                    symbol = ann.symbol[idx]
                    
                    # Map to AAMI class
                    if symbol not in self.config.AAMI_MAPPING:
                        continue
                    label = self.config.AAMI_MAPPING[symbol]
                    
                    # Check bounds
                    start = samp - before
                    end = samp + after
                    
                    if start < 0 or end > len(sig):
                        continue
                    
                    # Extract segment
                    segment = sig[start:end]
                    if len(segment) != seg_len:
                        continue
                    
                    segments.append(segment)
                    labels.append(label)
                
            except Exception as e:
                self._log(f"  Error processing {rec_name}: {e}")
                continue
        
        X = np.array(segments, dtype=np.float32)
        y = np.array(labels, dtype=np.int64)
        
        self._log(f"Extracted: {len(y)} segments")
        self._log(f"Distribution: {dict(Counter(y))}")
        
        return X, y


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Download and create ECG datasets")
    parser.add_argument('--download-all', action='store_true', 
                        help='Download MIT-BIH and INCART')
    parser.add_argument('--download-mitbih', action='store_true',
                        help='Download MIT-BIH only')
    parser.add_argument('--download-incart', action='store_true',
                        help='Download INCART only')
    parser.add_argument('--create-raw', action='store_true',
                        help='Create RAW datasets (no SMOTE)')
    parser.add_argument('--force', action='store_true',
                        help='Force re-download/recreate')
    
    args = parser.parse_args()
    
    if args.download_all or args.download_mitbih or args.download_incart:
        downloader = DatasetDownloader()
        if args.download_all or args.download_mitbih:
            downloader.download_mitbih(force=args.force)
        if args.download_all or args.download_incart:
            downloader.download_incart(force=args.force)
    
    if args.create_raw:
        creator = DatasetCreator()
        creator.create_all_raw()
    
    if not any([args.download_all, args.download_mitbih, 
                args.download_incart, args.create_raw]):
        parser.print_help()


if __name__ == '__main__':
    main()
