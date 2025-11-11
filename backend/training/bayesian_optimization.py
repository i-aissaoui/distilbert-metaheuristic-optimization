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
    
    def __init__(self, train_dataset, val_dataset, num_epochs=1, log_dir=None):
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
    
    def objective(self, trial):
        """
        Objective function for Optuna to optimize.
        
        Args:
            trial: Optuna trial object
        
        Returns:
            F1 score to maximize
        """
        self.trial_count += 1
        
        # Suggest hyperparameters (EXPANDED for better exploration)
        learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-4, log=True)
        batch_size = trial.suggest_int('batch_size', 4, 64, step=4)
        dropout = trial.suggest_float('dropout', 0.0, 0.5)
        frozen_layers = trial.suggest_int('frozen_layers', 0, 6)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Bayesian Optimization - Trial {self.trial_count}")
        logger.info(f"{'='*60}")
        logger.info(f"Learning Rate: {learning_rate:.6f}")
        logger.info(f"Batch Size: {batch_size}")
        logger.info(f"Dropout: {dropout:.3f}")
        logger.info(f"Frozen Layers: {frozen_layers}")
        
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
        logger.info(f"  Learning Rate: [1e-5, 5e-5] (log scale)")
        logger.info(f"  Batch Size: [8, 12, 16, 24, 32]")
        logger.info(f"  Dropout: [0.1, 0.3]")
        logger.info(f"  Frozen Layers: [0, 4]")
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42)
        )
        
        # Run optimization
        study.optimize(self.objective, n_trials=n_trials)
        
        # IMMEDIATELY update status to completed after optimization
        bayesian_results_dir = self.log_dir / "results" / "bayesian"
        bayesian_results_dir.mkdir(parents=True, exist_ok=True)
        completion_status = {
            'status': 'completed',
            'progress': 100,
            'trial': n_trials,
            'message': '✅ Bayesian optimization completed successfully!'
        }
        with open(bayesian_results_dir / 'status.json', 'w') as f:
            json.dump(completion_status, f, indent=2)
        logger.info("✅ Status immediately set to completed")
        
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
        logger.info(f"⏱️  Total optimization time: {optimization_time/60:.2f} minutes ({optimization_time:.1f} seconds)")
        logger.info(f"Best Configuration:")
        for key, value in best_trial.params.items():
            logger.info(f"  {key}: {value}")
        
        # Save final status
        final_status = {
            'status': 'completed',
            'trial': self.trial_count,
            'best_score': float(best_trial.value),
            'best_params': self.best_params,
            'total_trials': n_trials
        }
        bayesian_results_dir = self.log_dir / "results" / "bayesian"
        bayesian_results_dir.mkdir(parents=True, exist_ok=True)
        status_file = bayesian_results_dir / "status.json"
        with open(status_file, 'w') as f:
            json.dump(final_status, f, indent=2)
        
        # Save to latest.json
        latest_result = {
            'type': 'bayesian',
            'completedAt': datetime.now().isoformat(),
            'accuracy': float(self.best_params.get('accuracy', 0)),
            'f1_score': float(best_trial.value),
            'message': f'Bayesian Optimization completed - Best F1: {best_trial.value:.4f}',
            'best_params': self.best_params,
            'optimization_params': {'n_trials': n_trials},
            'training_time': float(optimization_time)  # in seconds
        }
        latest_file = bayesian_results_dir / "latest.json"
        with open(latest_file, 'w') as f:
            json.dump(latest_result, f, indent=2)
        logger.info(f"✅ Saved Bayesian results to latest.json")
        
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
    SUBSET_FRACTION = 0.05  # Use only 5% of data for training to show accuracy gap
    
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
    optimizer = BayesianOptimizer(
        train_dataset=train_dataset_subset,
        val_dataset=val_dataset_full,
        num_epochs=NUM_EPOCHS
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
