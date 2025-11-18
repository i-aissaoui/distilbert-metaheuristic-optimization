"""Bayesian Optimization using Optuna for hyperparameter tuning."""
import os
import warnings
warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import optuna
from optuna.samplers import TPESampler
import json
import sys
from pathlib import Path
import logging
from datetime import datetime
import numpy as np

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from training.data_loader import prepare_datasets
from training.trainer import DistilBERTTrainer, freeze_layers

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BayesianOptimizer:
    """Bayesian Optimization using Optuna for DistilBERT."""
    
    def __init__(self, train_dataset, val_dataset, num_epochs=1, log_dir=None, optimize_mask=None, fixed_values=None):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.num_epochs = num_epochs
        self.model_name = "distilbert-base-uncased"
        self.num_labels = 2  # Binary classification: spam/ham
        self.max_length = 128
        
        # Track best configuration
        self.best_score = 0
        self.best_params = None
        self.trial_count = 0
        
        # Real-time tracking
        self.history = []
        self.animation_data = []
        self.log_dir = log_dir or Path(__file__).parent.parent / "logs"
        self.log_dir.mkdir(exist_ok=True)
        
        # Optimization config
        self.param_names = ['learning_rate', 'batch_size', 'dropout', 'frozen_layers']
        default_fixed = {
            'learning_rate': 2e-5,
            'batch_size': 16,
            'dropout': 0.1,
            'frozen_layers': 0,
        }
        self.optimize_mask = optimize_mask or {
            'learning_rate': True,
            'batch_size': True,
            'dropout': True,
            'frozen_layers': True,
        }
        self.fixed_values = {**default_fixed, **(fixed_values or {})}
    
    def objective(self, trial):
        """
        Objective function for Optuna to optimize.
        
        Args:
            trial: Optuna trial object
        
        Returns:
            F1 score to maximize
        """
        self.trial_count += 1
        
        # Suggest hyperparameters (EXPANDED) with optimize/fixed handling
        learning_rate = (
            trial.suggest_float('learning_rate', 1e-6, 1e-4, log=True)
            if self.optimize_mask.get('learning_rate', True) else float(self.fixed_values['learning_rate'])
        )
        batch_size = (
            trial.suggest_int('batch_size', 4, 64, step=4)
            if self.optimize_mask.get('batch_size', True) else int(self.fixed_values['batch_size'])
        )
        dropout = (
            trial.suggest_float('dropout', 0.0, 0.5)
            if self.optimize_mask.get('dropout', True) else float(self.fixed_values['dropout'])
        )
        frozen_layers = (
            trial.suggest_int('frozen_layers', 0, 6)
            if self.optimize_mask.get('frozen_layers', True) else int(self.fixed_values['frozen_layers'])
        )
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Bayesian Optimization - Trial {self.trial_count}")
        logger.info(f"{'='*60}")
        lr_mode = "optimized" if self.optimize_mask.get('learning_rate', True) else f"fixed ({self.fixed_values.get('learning_rate')})"
        bs_mode = "optimized" if self.optimize_mask.get('batch_size', True) else f"fixed ({self.fixed_values.get('batch_size')})"
        dr_mode = "optimized" if self.optimize_mask.get('dropout', True) else f"fixed ({self.fixed_values.get('dropout')})"
        fr_mode = "optimized" if self.optimize_mask.get('frozen_layers', True) else f"fixed ({self.fixed_values.get('frozen_layers')})"
        logger.info(f"Learning Rate: {learning_rate:.6f} [{lr_mode}]")
        logger.info(f"Batch Size: {batch_size} [{bs_mode}]")
        logger.info(f"Dropout: {dropout:.3f} [{dr_mode}]")
        logger.info(f"Frozen Layers: {frozen_layers} [{fr_mode}]")
        
        try:
            # Initialize model with current hyperparameters (offline mode)
            import os
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
            from transformers import DistilBertConfig
            config = DistilBertConfig.from_pretrained(self.model_name, local_files_only=True)
            config.num_labels = self.num_labels
            config.dropout = dropout
            config.attention_dropout = dropout
            config.seq_classif_dropout = dropout
            
            model = DistilBertForSequenceClassification.from_pretrained(
                self.model_name,
                config=config,
                local_files_only=True
            )
            
            # Freeze layers if specified
            if frozen_layers > 0:
                freeze_layers(model, frozen_layers)
            
            # Initialize trainer
            trainer = DistilBERTTrainer(
                model=model,
                train_dataset=self.train_dataset,
                val_dataset=self.val_dataset,
                learning_rate=learning_rate,
                batch_size=batch_size,
                num_epochs=self.num_epochs
            )
            
            # Train and evaluate
            metrics = trainer.train()
            f1_score = metrics['f1_weighted']
            
            logger.info(f"F1 Score: {f1_score:.4f}")
            logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
            
            # Track best configuration
            is_best = f1_score > self.best_score
            if is_best:
                self.best_score = f1_score
                self.best_params = {
                    'learning_rate': learning_rate,
                    'batch_size': batch_size,
                    'dropout': dropout,
                    'frozen_layers': frozen_layers,
                    'f1_score': f1_score,
                    'accuracy': metrics['accuracy']
                }
                logger.info(f"🎉 New best configuration found!")
            
            # Save trial data for animation
            trial_data = {
                'trial': self.trial_count,
                'learning_rate': float(learning_rate),
                'batch_size': int(batch_size),
                'dropout': float(dropout),
                'frozen_layers': int(frozen_layers),
                'f1_score': float(f1_score),
                'accuracy': float(metrics['accuracy']),
                'is_best': is_best
            }
            self.animation_data.append(trial_data)
            
            # Save to file for real-time access
            self._save_progress()
            
            return f1_score
        
        except Exception as e:
            logger.error(f"Error in trial {self.trial_count}: {e}")
            return 0.0
    
    def _save_progress(self):
        """Save current progress to JSON files for real-time access."""
        def convert_to_native(obj):
            """Convert numpy types to Python native types."""
            import numpy as np
            if isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_native(item) for item in obj]
            elif isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        try:
            # Create results directory
            bayesian_results_dir = self.log_dir / "results" / "bayesian"
            bayesian_results_dir.mkdir(parents=True, exist_ok=True)
            
            # Save animation data
            animation_file = bayesian_results_dir / "animation.json"
            with open(animation_file, 'w') as f:
                json.dump({'data': convert_to_native(self.animation_data)}, f, indent=2)
            
            # Save history data
            if self.best_params:
                history_entry = {
                    'trial': self.trial_count,
                    'best_score': float(self.best_score),
                    'best_params': convert_to_native(self.best_params)
                }
                self.history.append(history_entry)
                
                history_file = bayesian_results_dir / "history.json"
                with open(history_file, 'w') as f:
                    json.dump({'data': convert_to_native(self.history)}, f, indent=2)
            
            # Save current status
            status = {
                'status': 'running',
                'trial': self.trial_count,
                'best_score': float(self.best_score) if self.best_score > 0 else None,
                'best_params': convert_to_native(self.best_params) if self.best_params else None,
                'total_trials': self.trial_count
            }
            status_file = bayesian_results_dir / "status.json"
            with open(status_file, 'w') as f:
                json.dump(status, f, indent=2)

            # Also update latest.json to reflect the most recent trial metrics (last trial wins)
            try:
                last_f1 = float(self.best_score) if self.best_score else 0.0
                last_acc = float(self.best_params.get('accuracy', 0)) if self.best_params else 0.0
                if self.animation_data:
                    last_entry = self.animation_data[-1]
                    last_f1 = float(last_entry.get('f1_score', last_f1))
                    last_acc = float(last_entry.get('accuracy', last_acc))
                latest_result = {
                    'type': 'bayesian',
                    'completedAt': datetime.now().isoformat(),
                    'accuracy': last_acc,
                    'f1_score': last_f1,
                    'message': 'Bayesian Optimization - trial update',
                    'best_params': convert_to_native(self.best_params) if self.best_params else {},
                    'optimization_params': {'n_trials': status.get('total_trials', self.trial_count)},
                    'training_time': 0.0,
                    'parameter_selection': {
                        'optimize': self.optimize_mask,
                        'fixed': self.fixed_values
                    },
                    'full_data_f1_score': 0.0,
                    'full_data_accuracy': 0.0,
                    'best_score': float(self.best_score) if self.best_score else 0.0
                }
                with open(bayesian_results_dir / 'latest.json', 'w') as f:
                    json.dump(latest_result, f, indent=2)
            except Exception:
                pass
                
        except Exception as e:
            logger.error(f"Error saving progress: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def optimize(self, n_trials=10):
        """
        Run Bayesian Optimization using Optuna.
        
        Args:
            n_trials: Number of optimization trials
        
        Returns:
            Best hyperparameters found
        """
        import time
        start_time = time.time()
        
        logger.info("\n" + "="*60)
        logger.info("Starting Bayesian Optimization (Optuna)")
        logger.info("="*60)
        logger.info(f"Number of Trials: {n_trials}")
        logger.info(f"Sampler: TPE (Tree-structured Parzen Estimator)")
        logger.info(f"Search Space:")
        logger.info(f"  Learning Rate: [1e-6, 1e-4] (log scale)")
        logger.info(f"  Batch Size: [4, 64]")
        logger.info(f"  Dropout: [0.0, 0.5]")
        logger.info(f"  Frozen Layers: [0, 6]")
        logger.info("Starting Parameter Modes:")
        for name in self.param_names:
            mode_str = "optimized" if self.optimize_mask.get(name, True) else f"fixed ({self.fixed_values.get(name)})"
            logger.info(f"Starting - {name}: {mode_str}")
        
        # Initialize latest.json to indicate a new run (prevents stale values)
        try:
            bayesian_results_dir = self.log_dir / "results" / "bayesian"
            bayesian_results_dir.mkdir(parents=True, exist_ok=True)
            init_latest = {
                'type': 'bayesian',
                'completedAt': datetime.now().isoformat(),
                'accuracy': 0.0,
                'f1_score': 0.0,
                'message': 'Bayesian Optimization initializing...',
                'best_params': {},
                'optimization_params': {'n_trials': n_trials},
                'training_time': 0.0,
                'parameter_selection': {
                    'optimize': self.optimize_mask,
                    'fixed': self.fixed_values
                },
                'full_data_f1_score': 0.0,
                'full_data_accuracy': 0.0
            }
            with open(bayesian_results_dir / 'latest.json', 'w') as f:
                json.dump(init_latest, f, indent=2)
        except Exception:
            pass

        # Create study
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42)
        )
        
        # Run optimization
        study.optimize(self.objective, n_trials=n_trials)
        
        # Get best results
        best_trial = study.best_trial
        self.best_params = {
            'learning_rate': best_trial.params['learning_rate'],
            'batch_size': best_trial.params['batch_size'],
            'dropout': best_trial.params['dropout'],
            'frozen_layers': best_trial.params['frozen_layers'],
            'f1_score': best_trial.value,
            'accuracy': self.best_params.get('accuracy', 0) if self.best_params else 0
        }
        
        # Calculate total optimization time
        end_time = time.time()
        optimization_time = end_time - start_time
        
        logger.info("\n" + "="*60)
        logger.info("Bayesian Optimization Complete!")
        logger.info("="*60)
        logger.info(f"Best F1 Score: {best_trial.value:.4f}")
        try:
            logger.info(f"Best Accuracy: {float(self.best_params.get('accuracy', 0)):.4f}")
        except Exception:
            logger.info("Best Accuracy: 0.0000")
        logger.info(f"⏱️  Total optimization time: {optimization_time/60:.2f} minutes ({optimization_time:.1f} seconds)")
        logger.info(f"Best Configuration:")
        for key, value in best_trial.params.items():
            logger.info(f"  {key}: {value}")
        # Explicit single-line result for terminal filters
        try:
            logger.info(
                f"BAYES RESULT -> F1: {best_trial.value:.4f}, Acc: {float(self.best_params.get('accuracy', 0)):.4f}, "
                f"Params: lr={best_trial.params.get('learning_rate')}, bs={best_trial.params.get('batch_size')}, "
                f"dropout={best_trial.params.get('dropout')}, frozen={best_trial.params.get('frozen_layers')}"
            )
        except Exception:
            pass
        
        # Save final status
        final_status = {
            'status': 'completed',
            'trial': self.trial_count,
            'best_score': float(best_trial.value),
            'best_params': self.best_params,
            'best_accuracy': float(self.best_params.get('accuracy', 0)) if self.best_params else 0.0,
            'total_trials': n_trials
        }
        bayesian_results_dir = self.log_dir / "results" / "bayesian"
        bayesian_results_dir.mkdir(parents=True, exist_ok=True)
        status_file = bayesian_results_dir / "status.json"
        with open(status_file, 'w') as f:
            json.dump(final_status, f, indent=2)

        # Determine last trial metrics (use last appended animation data)
        last_f1 = float(best_trial.value)
        last_acc = float(self.best_params.get('accuracy', 0)) if self.best_params else 0.0
        if self.animation_data:
            try:
                last_entry = self.animation_data[-1]
                last_f1 = float(last_entry.get('f1_score', last_f1))
                last_acc = float(last_entry.get('accuracy', last_acc))
            except Exception:
                pass

        # Save to latest.json for frontend consumption (primary metrics = last trial)
        latest_result = {
            'type': 'bayesian',
            'completedAt': datetime.now().isoformat(),
            'accuracy': last_acc,
            'f1_score': last_f1,
            'message': f'Bayesian Optimization completed - Best F1: {best_trial.value:.4f}',
            'best_params': self.best_params or {},
            'optimization_params': {'n_trials': n_trials},
            'training_time': float(optimization_time),
            'parameter_selection': {
                'optimize': self.optimize_mask,
                'fixed': self.fixed_values
            },
            # Placeholders indicating 100% full-data training not yet performed
            'full_data_f1_score': 0.0,
            'full_data_accuracy': 0.0,
            # Also include best-of-run for reference
            'best_score': float(best_trial.value)
        }
        latest_file = bayesian_results_dir / "latest.json"
        with open(latest_file, 'w') as f:
            json.dump(latest_result, f, indent=2)
        logger.info(f"✅ Saved Bayesian results to latest.json")
        # Final explicit saved line
        try:
            logger.info(
                f"BAYES SAVED -> latest.json | Best F1: {best_trial.value:.4f}, Acc: {float(self.best_params.get('accuracy', 0)):.4f}"
            )
        except Exception:
            pass

        return self.best_params


def run_bayesian_optimization():
    """Main function to run Bayesian optimization."""
    
    # Get parameters from command line or use defaults
    import sys
    N_TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    
    # Configuration
    MODEL_NAME = "distilbert-base-uncased"
    MAX_LENGTH = 128
    NUM_EPOCHS = 1  # 1 epoch to prevent overfitting on small data
    
    logger.info("Loading dataset...")
    SUBSET_FRACTION = 0.20  # Use 10% of data for training
    
    # Load tokenizer
    from transformers import DistilBertTokenizer
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    
    # Get full datasets for validation
    train_dataset_full, val_dataset_full, test_dataset_full = prepare_datasets(
        tokenizer, 
        MAX_LENGTH,
        subset_fraction=1.0
    )
    
    # Get small subset for training only
    train_dataset_subset, _, _ = prepare_datasets(
        tokenizer, 
        MAX_LENGTH,
        subset_fraction=SUBSET_FRACTION
    )
    
    logger.info(f"⚠️  Training on only {SUBSET_FRACTION*100:.0f}% of data, testing on 100% to demonstrate generalization gap")
    logger.info(f"Train size: {len(train_dataset_subset)}, Val size: {len(val_dataset_full)}")
    
    logger.info("Initializing Bayesian optimizer...")
    # Load optimization config (new path with legacy fallback)
    config_path = Path(__file__).parent.parent / "config" / "optimization_config.json"
    legacy_config_path = Path(__file__).parent.parent / "logs" / "settings" / "optimization_config.json"
    optimize = {
        'learning_rate': True,
        'batch_size': True,
        'dropout': True,
        'frozen_layers': True,
    }
    fixed = {
        'learning_rate': 2e-5,
        'batch_size': 16,
        'dropout': 0.1,
        'frozen_layers': 0,
    }
    try:
        loaded = None
        if config_path.exists():
            with open(config_path, 'r') as f:
                loaded = json.load(f)
        elif legacy_config_path.exists():
            with open(legacy_config_path, 'r') as f:
                loaded = json.load(f)
        if loaded:
            data = loaded
            fixed_keys_cfg = set()
            if isinstance(data.get('fixed'), dict):
                for k, v in data['fixed'].items():
                    if v is not None:
                        fixed[k] = v
                        fixed_keys_cfg.add(k)
            if isinstance(data.get('optimize'), dict):
                for k, v in data['optimize'].items():
                    if v is not None:
                        optimize[k] = bool(v)
            for name in fixed_keys_cfg:
                optimize[name] = False
    except Exception as e:
        logger.warning(f"Could not load optimization config: {e}")

    optimizer = BayesianOptimizer(
        train_dataset=train_dataset_subset,
        val_dataset=val_dataset_full,
        num_epochs=NUM_EPOCHS,
        optimize_mask=optimize,
        fixed_values=fixed
    )
    
    logger.info("Starting optimization...")
    best_params = optimizer.optimize(n_trials=N_TRIALS)
    
    # Save best hyperparameters
    output_file = Path(__file__).parent.parent / "models" / "bayesian_best_hyperparameters.json"
    output_file.parent.mkdir(exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(best_params, f, indent=2)
    
    logger.info(f"\nBest hyperparameters saved to: {output_file}")
    logger.info("Optimization complete!")


if __name__ == "__main__":
    run_bayesian_optimization()
