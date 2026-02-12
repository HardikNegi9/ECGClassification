#!/usr/bin/env python3
"""
ECG Classification Training Script

Main entry point for training models from the command line.

Usage:
    # Train with config file
    python scripts/train.py --config configs/paper1_inceptiontime.yaml
    
    # Train with command line overrides
    python scripts/train.py --config configs/paper1_inceptiontime.yaml \
        --model.variant large --training.epochs 150
    
    # K-Fold cross-validation
    python scripts/train.py --config configs/paper1_inceptiontime.yaml --kfold
    
    # Quick test run
    python scripts/train.py --config configs/paper1_inceptiontime.yaml --debug
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np

from src.data import ECGDataModule
from src.models import create_model
from src.training import Trainer, KFoldTrainer
from src.evaluation import PaperMetrics, MetricsVisualizer, InferenceTimer, evaluate_model
from src.utils import (
    ExperimentConfig, load_config, set_seed, get_device, 
    get_gpu_info, setup_logging, ExperimentTracker, 
    count_parameters, print_model_summary
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train ECG classification models',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Config
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config YAML file')
    
    # Overrides
    parser.add_argument('--name', type=str, default=None,
                       help='Experiment name (overrides config)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed')
    parser.add_argument('--device', type=str, default=None,
                       help='Device (auto, cuda, cuda:0, cpu)')
    
    # Data
    parser.add_argument('--data.mode', type=str, default=None, dest='data_mode',
                       help='Data mode (combined, mitbih, incart)')
    parser.add_argument('--data.batch_size', type=int, default=None, dest='data_batch_size',
                       help='Batch size')
    parser.add_argument('--data.samples_per_class', type=int, default=None, dest='data_samples_per_class',
                       help='Samples per class (e.g., 10000 = 50k total, None = use all)')
    
    # Model
    parser.add_argument('--model.name', type=str, default=None, dest='model_name',
                       help='Model name')
    parser.add_argument('--model.variant', type=str, default=None, dest='model_variant',
                       help='Model variant (tiny, base, large)')
    
    # Training
    parser.add_argument('--training.epochs', type=int, default=None, dest='training_epochs',
                       help='Number of epochs')
    parser.add_argument('--training.lr', type=float, default=None, dest='training_lr',
                       help='Learning rate')
    
    # K-Fold
    parser.add_argument('--kfold', action='store_true',
                       help='Enable K-Fold cross-validation')
    parser.add_argument('--kfold.n_splits', type=int, default=None, dest='kfold_n_splits',
                       help='Number of folds')
    
    # Debug
    parser.add_argument('--debug', action='store_true',
                       help='Quick debug run (1 epoch, no checkpointing)')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='Verbose output')
    
    return parser.parse_args()


def apply_overrides(config: ExperimentConfig, args):
    """Apply command-line overrides to config."""
    if args.name:
        config.name = args.name
    if args.seed:
        config.seed = args.seed
    if args.device:
        config.device = args.device
    
    if args.data_mode:
        config.data.mode = args.data_mode
    if args.data_batch_size:
        config.data.batch_size = args.data_batch_size
    if args.data_samples_per_class:
        config.data.samples_per_class = args.data_samples_per_class
    
    if args.model_name:
        config.model.name = args.model_name
    if args.model_variant:
        config.model.variant = args.model_variant
    
    if args.training_epochs:
        config.training.epochs = args.training_epochs
    if args.training_lr:
        config.training.lr = args.training_lr
    
    if args.kfold:
        config.kfold.enabled = True
    if args.kfold_n_splits:
        config.kfold.n_splits = args.kfold_n_splits
    
    if args.debug:
        config.training.epochs = 1
        config.training.patience = 1
        config.name = f"{config.name}_debug"
    
    return config


def main():
    args = parse_args()
    
    # Load config
    print(f"\nLoading config: {args.config}")
    config = load_config(args.config)
    config = apply_overrides(config, args)
    
    # Setup
    set_seed(config.seed)
    device = get_device(config.device)
    logger = setup_logging(config.log_dir, config.name)
    
    print(f"\n{'='*70}")
    print(f"ECG CLASSIFICATION TRAINING")
    print(f"{'='*70}")
    print(f"Experiment: {config.name}")
    print(f"Device: {device}")
    print(f"Seed: {config.seed}")
    
    # GPU info
    gpu_info = get_gpu_info()
    if gpu_info['cuda_available']:
        for i, dev in enumerate(gpu_info['devices']):
            print(f"GPU {i}: {dev['name']} ({dev['total_memory_gb']:.1f} GB)")
    
    # Experiment tracking
    tracker = ExperimentTracker(config.name, config.experiment_dir)
    tracker.log_config(config)
    
    # Load data
    print(f"\n{'='*70}")
    print("LOADING DATA")
    print(f"{'='*70}")
    
    # Debug: Print actual config values
    print(f"Config values:")
    print(f"  mode: {config.data.mode}")
    print(f"  batch_size: {config.data.batch_size}")
    print(f"  num_workers: {config.data.num_workers}")
    print(f"  samples_per_class: {getattr(config.data, 'samples_per_class', None)}")
    print(f"  model: {config.model.name} ({config.model.variant})")
    
    data_module = ECGDataModule(
        mode=config.data.mode,
        balancing_method=getattr(config.data, 'balancing_method', 'smote'),
        samples_per_class=getattr(config.data, 'samples_per_class', None),
        test_size=config.data.test_size,
        val_size=config.data.val_size,
        seed=config.data.seed,
        verbose=args.verbose
    )
    
    # Create model
    print(f"\n{'='*70}")
    print("CREATING MODEL")
    print(f"{'='*70}")
    
    model = create_model(
        name=config.model.name,
        num_classes=config.model.num_classes,
        variant=config.model.variant,
        compile=config.model.use_compile
    )
    model = model.to(device)
    
    print_model_summary(model)
    
    # Training
    if config.kfold.enabled:
        # K-Fold Cross-Validation
        print(f"\n{'='*70}")
        print(f"K-FOLD CROSS-VALIDATION ({config.kfold.n_splits} folds)")
        print(f"{'='*70}")
        
        X, y = data_module.get_kfold_data()
        
        def model_fn():
            return create_model(
                name=config.model.name,
                num_classes=config.model.num_classes,
                variant=config.model.variant,
                compile=False  # Don't compile for K-fold (creates new models)
            )
        
        kfold_trainer = KFoldTrainer(
            model_fn=model_fn,
            X=X,
            y=y,
            n_splits=config.kfold.n_splits,
            apply_smote=config.kfold.apply_smote_per_fold,
            smote_target=getattr(config.kfold, 'smote_target', None) or getattr(config.data, 'smote_target', None),
            batch_size=config.data.batch_size,
            epochs=config.training.epochs,
            patience=config.training.patience,
            lr_patience=getattr(config.training, 'lr_patience', 5),
            lr_reduce_factor=getattr(config.training, 'lr_reduce_factor', 0.5),
            min_lr=getattr(config.training, 'min_lr', 1e-7),
            max_lr_reductions=getattr(config.training, 'max_lr_reductions', 3),
            device=device,
            experiment_name=config.name,
            use_scalograms=(config.model.name == 'efficientnet_scalogram'),
            scalogram_img_size=getattr(config.model, 'img_size', 224)
        )
        
        results = kfold_trainer.run(verbose=args.verbose)
        
        # Log results
        tracker.log_metrics({
            'accuracy_mean': results['mean_accuracy'],
            'accuracy_std': results['std_accuracy'],
            'f1_mean': results['mean_f1'],
            'f1_std': results['std_f1'],
        })
        
        tracker.save_results(results)
        kfold_trainer.save_results(str(tracker.dir))
        
    else:
        # Standard training
        print(f"\n{'='*70}")
        print("TRAINING")
        print(f"{'='*70}")
        
        # Get loaders
        if config.model.name == 'efficientnet_scalogram':
            loaders = data_module.get_scalogram_loaders(
                batch_size=config.data.batch_size,
                num_workers=config.data.num_workers
            )
            print(f"  Scalogram dataloaders created: train={len(loaders['train'].dataset)}, val={len(loaders['val'].dataset)}, test={len(loaders['test'].dataset)}")
        else:
            loaders = data_module.get_signal_loaders(
                batch_size=config.data.batch_size,
                num_workers=config.data.num_workers
            )
        
        print("\nInitializing Trainer...")
        
        # Train
        trainer = Trainer(
            model=model,
            train_loader=loaders['train'],
            val_loader=loaders['val'],
            device=device,
            lr=config.training.lr,
            weight_decay=config.training.weight_decay,
            use_amp=config.training.use_amp,
            use_class_weights=config.training.use_class_weights,
            checkpoint_dir=config.checkpoint_dir
        )
        
        history = trainer.train(
            epochs=config.training.epochs,
            patience=config.training.patience,
            lr_patience=getattr(config.training, 'lr_patience', 5),
            warmup_epochs=config.training.warmup_epochs,
            lr_reduce_factor=getattr(config.training, 'lr_reduce_factor', 0.5),
            min_lr=getattr(config.training, 'min_lr', 1e-7),
            max_lr_reductions=getattr(config.training, 'max_lr_reductions', 3),
            verbose=args.verbose
        )
        
        # Evaluate on test set
        print(f"\n{'='*70}")
        print("EVALUATION")
        print(f"{'='*70}")
        
        eval_results = evaluate_model(model, loaders['test'], device)
        metrics = eval_results['metrics']
        metrics.print_full_report()
        
        # ================================================================
        # SAVE ALL PAPER DATA
        # ================================================================
        print(f"\n{'='*70}")
        print("SAVING PAPER DATA")
        print(f"{'='*70}")
        
        # 1. Predictions (for reproducibility)
        np.save(tracker.dir / 'y_true.npy', eval_results['y_true'])
        np.save(tracker.dir / 'y_pred.npy', eval_results['y_pred'])
        np.save(tracker.dir / 'y_probs.npy', eval_results['y_probs'])
        print("  [+] Saved predictions (y_true, y_pred, y_probs)")
        
        # 2. LaTeX tables
        latex_per_class = metrics.export_latex_table(str(tracker.dir / 'per_class_metrics.tex'))
        print("  [+] Saved LaTeX table: per_class_metrics.tex")
        
        # 3. CSV files
        metrics.export_csv(str(tracker.dir / 'per_class_metrics.csv'))
        print("  [+] Saved CSV: per_class_metrics.csv")
        
        # 4. Visualizations
        visualizer = MetricsVisualizer(metrics)
        
        fig_cm = visualizer.plot_confusion_matrix()
        tracker.save_figure(fig_cm, 'confusion_matrix.png')
        
        fig_cm_norm = visualizer.plot_confusion_matrix(normalize=True)
        tracker.save_figure(fig_cm_norm, 'confusion_matrix_normalized.png')
        
        fig_pc = visualizer.plot_per_class_metrics()
        tracker.save_figure(fig_pc, 'per_class_metrics.png')
        
        if metrics.y_probs is not None:
            fig_roc = visualizer.plot_roc_curves()
            tracker.save_figure(fig_roc, 'roc_curves.png')
        
        fig_lc = visualizer.plot_learning_curves(history)
        tracker.save_figure(fig_lc, 'learning_curves.png')
        print("  [+] Saved all figures (confusion matrices, ROC, learning curves)")
        
        # 5. Inference timing
        input_shape = (1, 3, 224, 224) if config.model.name == 'efficientnet_scalogram' else (1, 360)
        timer = InferenceTimer(model, device)
        timing = timer.measure_single(input_shape=input_shape)
        print(f"  [+] Inference time: {timing['mean_ms']:.2f} ms/sample ({timing['throughput_per_sec']:.0f} samples/sec)")
        
        # 6. Model info
        n_params = count_parameters(model)
        training_time = history.get('training_time_seconds', 0)
        best_epoch = history.get('best_epoch', len(history.get('val_loss', [])))  
        best_val_loss = min(history.get('val_loss', [0]))
        
        # 7. Dataset statistics
        dataset_stats = {
            'mode': config.data.mode,
            'balancing_method': getattr(config.data, 'balancing_method', 'smote'),
            'samples_per_class': getattr(config.data, 'samples_per_class', None),
            'train_samples': len(loaders['train'].dataset),
            'val_samples': len(loaders['val'].dataset),
            'test_samples': len(loaders['test'].dataset),
            'total_samples': len(loaders['train'].dataset) + len(loaders['val'].dataset) + len(loaders['test'].dataset),
        }
        print(f"  [+] Dataset: {dataset_stats['total_samples']} total samples")
        
        # Log metrics and save models
        tracker.log_metrics(metrics.overall)
        tracker.save_model(model, 'best_model.pt')  # Save best model (loaded by early stopping)
        
        # Also copy checkpoint if it exists
        checkpoint_path = Path(config.checkpoint_dir) / 'best_model.pt'
        if checkpoint_path.exists():
            import shutil
            shutil.copy(checkpoint_path, tracker.dir / 'best_model_checkpoint.pt')
        print("  [+] Saved best model")
        
        # 8. Complete results JSON
        tracker.save_results({
            # Training history
            'history': history,
            'best_epoch': best_epoch,
            'best_val_loss': best_val_loss,
            'total_epochs': len(history.get('train_loss', [])),
            'training_time_seconds': training_time,
            
            # Classification metrics  
            'overall_metrics': metrics.overall,
            'per_class_metrics': {k: v for k, v in metrics.per_class.items()},
            'confusion_matrix': metrics.cm.tolist(),
            
            # Model info
            'model_name': config.model.name,
            'model_variant': config.model.variant,
            'num_parameters': n_params,
            
            # Inference timing
            'inference_timing': timing,
            
            # Dataset info
            'dataset': dataset_stats,
            
            # Config
            'config': config.to_dict() if hasattr(config, 'to_dict') else str(config),
        })
        print("  [+] Saved complete results.json")
        
        # 9. Generate paper-ready summary markdown
        summary_md = f"""# {config.name} - Results Summary

## Overall Performance

| Metric | Value |
|--------|-------|
| Accuracy | {metrics.overall['accuracy']:.4f} |
| F1 (Macro) | {metrics.overall['f1_macro']:.4f} |
| F1 (Weighted) | {metrics.overall['f1_weighted']:.4f} |
| AUC-ROC | {metrics.overall.get('auc_macro', 'N/A')} |
| Cohen's Kappa | {metrics.overall['cohen_kappa']:.4f} |
| MCC | {metrics.overall['mcc']:.4f} |

## Per-Class Performance

| Class | Precision | Recall | Specificity | F1 |
|-------|-----------|--------|-------------|----|
""" + '\n'.join([
            f"| {c} | {metrics.per_class[c]['precision']:.4f} | {metrics.per_class[c]['recall']:.4f} | {metrics.per_class[c]['specificity']:.4f} | {metrics.per_class[c]['f1']:.4f} |"
            for c in ['N', 'S', 'V', 'F', 'Q']
        ]) + f"""

## Training Details

- **Model:** {config.model.name} ({config.model.variant})
- **Parameters:** {n_params:,}
- **Best Epoch:** {best_epoch}
- **Training Time:** {training_time/60:.1f} min
- **Inference:** {timing['mean_ms']:.2f} ms/sample

## Dataset

- **Mode:** {dataset_stats['mode']}
- **Balancing:** {dataset_stats['balancing_method']}
- **Train:** {dataset_stats['train_samples']:,} samples
- **Val:** {dataset_stats['val_samples']:,} samples  
- **Test:** {dataset_stats['test_samples']:,} samples
"""
        with open(tracker.dir / 'RESULTS_SUMMARY.md', 'w') as f:
            f.write(summary_md)
        print("  [+] Saved RESULTS_SUMMARY.md")
    
    # Finalize
    tracker.finalize()
    
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"Results saved to: {tracker.dir}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
