#!/usr/bin/env python3
"""
ECG Classification Evaluation Script

Evaluate trained models and generate paper-ready metrics and figures.

Usage:
    # Evaluate a trained model
    python scripts/evaluate.py --model-path checkpoints/paper1_inceptiontime/best_model.pt \
        --config configs/paper1_inceptiontime.yaml
    
    # Evaluate with test-time augmentation
    python scripts/evaluate.py --model-path checkpoints/model.pt --tta
    
    # Compare multiple models
    python scripts/evaluate.py --compare \
        experiments/paper1_inceptiontime/final_model.pt \
        experiments/paper2_efficientnet/final_model.pt \
        experiments/paper3_nsht/final_model.pt
    
    # Measure inference time
    python scripts/evaluate.py --model-path checkpoints/model.pt --benchmark
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np

from src.data import ECGDataModule
from src.models import create_model
from src.evaluation import (
    PaperMetrics, MetricsVisualizer, InferenceTimer,
    evaluate_model, compare_models
)
from src.utils import (
    ExperimentConfig, load_config, set_seed, get_device
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate ECG classification models',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--model-path', type=str,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str,
                       help='Path to config YAML file')
    
    # Evaluation options
    parser.add_argument('--tta', action='store_true',
                       help='Test-time augmentation')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run inference benchmarking')
    
    # Model comparison
    parser.add_argument('--compare', nargs='+', type=str,
                       help='Compare multiple model paths')
    
    # Output
    parser.add_argument('--output-dir', type=str, default='evaluation_results',
                       help='Output directory')
    parser.add_argument('--latex', action='store_true',
                       help='Generate LaTeX tables')
    
    # Device
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (auto, cuda, cpu)')
    
    return parser.parse_args()


def evaluate_single(model_path: str, config_path: str, device: torch.device,
                    output_dir: Path, benchmark: bool = False, latex: bool = False):
    """Evaluate a single model."""
    print(f"\n{'='*70}")
    print(f"EVALUATING: {model_path}")
    print(f"{'='*70}")
    
    # Load config
    config = load_config(config_path)
    
    # Load data
    data_module = ECGDataModule(
        mode=config.data.mode,
        balancing_method=getattr(config.data, 'balancing_method', 'smote'),
        samples_per_class=getattr(config.data, 'samples_per_class', None),
        test_size=config.data.test_size,
        val_size=config.data.val_size,
        seed=config.data.seed,
        verbose=True
    )
    
    # Get test loader
    if config.model.name == 'efficientnet_scalogram':
        loaders = data_module.get_scalogram_loaders(batch_size=256, num_workers=0)
    else:
        loaders = data_module.get_signal_loaders(batch_size=2048, num_workers=4)
    
    # Create and load model
    model = create_model(
        name=config.model.name,
        num_classes=config.model.num_classes,
        variant=config.model.variant,
        compile=False
    )
    
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    
    # Evaluate
    results = evaluate_model(model, loaders['test'], device)
    metrics = results['metrics']
    
    # Print report
    metrics.print_full_report()
    
    # Save outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ================================================================
    # SAVE ALL PAPER DATA
    # ================================================================
    print(f"\nSaving paper data to: {output_dir}")
    
    # 1. Predictions (for reproducibility)
    np.save(output_dir / 'y_true.npy', results['y_true'])
    np.save(output_dir / 'y_pred.npy', results['y_pred'])  
    np.save(output_dir / 'y_probs.npy', results['y_probs'])
    print("  [+] Saved predictions (y_true, y_pred, y_probs)")
    
    # 2. Visualizations
    visualizer = MetricsVisualizer(metrics)
    
    fig_cm = visualizer.plot_confusion_matrix()
    fig_cm.savefig(output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    
    fig_cm_norm = visualizer.plot_confusion_matrix(normalize=True)
    fig_cm_norm.savefig(output_dir / 'confusion_matrix_normalized.png', dpi=300, bbox_inches='tight')
    
    fig_pc = visualizer.plot_per_class_metrics()
    fig_pc.savefig(output_dir / 'per_class_metrics.png', dpi=300, bbox_inches='tight')
    
    if metrics.y_probs is not None:
        fig_roc = visualizer.plot_roc_curves()
        fig_roc.savefig(output_dir / 'roc_curves.png', dpi=300, bbox_inches='tight')
    
    # Export metrics
    metrics.export_csv(str(output_dir / 'metrics.csv'))
    
    if latex:
        metrics.export_latex_table(str(output_dir / 'metrics.tex'))
        print(f"\nLaTeX table saved to: {output_dir / 'metrics.tex'}")
    
    # Benchmarking
    if benchmark:
        print(f"\n{'='*70}")
        print("INFERENCE BENCHMARKING")
        print(f"{'='*70}")
        
        timer = InferenceTimer(model, device)
        
        # Single sample timing
        single_timing = timer.measure_single(n_runs=100)
        print(f"\nSingle Sample Inference:")
        print(f"  Mean: {single_timing['mean_ms']:.3f} ms")
        print(f"  Std:  {single_timing['std_ms']:.3f} ms")
        print(f"  Throughput: {single_timing['throughput_per_sec']:.0f} samples/sec")
        
        # Batch timing
        batch_timing = timer.measure_batch()
        print(f"\nBatch Inference:")
        print(batch_timing.to_string(index=False))
        
        # Save timing
        batch_timing.to_csv(output_dir / 'inference_timing.csv', index=False)
    else:
        single_timing = None
    
    # 3. Save complete results.json
    import json
    results_dict = {
        'model_path': str(model_path),
        'config_path': str(config_path),
        'overall_metrics': metrics.overall,
        'per_class_metrics': {k: v for k, v in metrics.per_class.items()},
        'confusion_matrix': metrics.cm.tolist(),
    }
    if single_timing:
        results_dict['inference_timing'] = single_timing
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results_dict, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else str(x))
    print("  [+] Saved results.json")
    
    print(f"\nResults saved to: {output_dir}")
    
    return results


def compare_multiple(model_paths: list, output_dir: Path, device: torch.device):
    """Compare multiple models."""
    print(f"\n{'='*70}")
    print("COMPARING MODELS")
    print(f"{'='*70}")
    
    # This requires model configs to be discoverable from paths
    # For now, assume all models use the same data config
    
    results = {}
    
    # Try to infer config from model path
    for model_path in model_paths:
        model_path = Path(model_path)
        model_name = model_path.parent.name
        
        # Try to find config
        config_path = None
        for config_file in Path('configs').glob('*.yaml'):
            if model_name in config_file.stem or config_file.stem.split('_')[1] in model_name:
                config_path = config_file
                break
        
        if config_path is None:
            print(f"Warning: Could not find config for {model_path}, skipping")
            continue
        
        sub_output = output_dir / model_name
        result = evaluate_single(
            str(model_path), str(config_path), device, sub_output
        )
        results[model_name] = result
    
    # Compare
    if len(results) > 1:
        comparison_df = compare_models(results)
        print("\n" + "="*70)
        print("MODEL COMPARISON")
        print("="*70)
        print(comparison_df.to_string(index=False))
        
        comparison_df.to_csv(output_dir / 'model_comparison.csv', index=False)
        print(f"\nComparison saved to: {output_dir / 'model_comparison.csv'}")


def main():
    args = parse_args()
    device = get_device(args.device)
    output_dir = Path(args.output_dir)
    
    if args.compare:
        compare_multiple(args.compare, output_dir, device)
    elif args.model_path and args.config:
        evaluate_single(
            args.model_path, args.config, device, output_dir,
            benchmark=args.benchmark, latex=args.latex
        )
    else:
        print("Usage: Provide --model-path and --config, or --compare with model paths")
        sys.exit(1)


if __name__ == '__main__':
    main()
