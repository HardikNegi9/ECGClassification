#!/usr/bin/env python3
"""
Quick Start Script

Trains all three paper models sequentially with default configs.

Usage:
    python scripts/train_all.py
    python scripts/train_all.py --kfold  # For paper experiments
"""

import subprocess
import sys
from pathlib import Path

CONFIGS = [
    'configs/paper1_inceptiontime.yaml',
    'configs/paper2_efficientnet.yaml',
    'configs/paper3_nsht.yaml',
]


def main():
    kfold = '--kfold' in sys.argv
    
    print("="*70)
    print("TRAINING ALL PAPER MODELS")
    print("="*70)
    
    for config in CONFIGS:
        print(f"\n{'='*70}")
        print(f"Training: {config}")
        print("="*70)
        
        cmd = [sys.executable, 'scripts/train.py', '--config', config]
        if kfold:
            cmd.append('--kfold')
        
        result = subprocess.run(cmd)
        
        if result.returncode != 0:
            print(f"Error training {config}")
            sys.exit(1)
    
    print("\n" + "="*70)
    print("ALL MODELS TRAINED SUCCESSFULLY")
    print("="*70)


if __name__ == '__main__':
    main()
