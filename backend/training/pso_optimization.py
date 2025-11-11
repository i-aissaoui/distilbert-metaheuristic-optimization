"""PSO-based hyperparameter optimization for DistilBERT."""
import os
import warnings
warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from pyswarm import pso
import json
import sys
from pathlib import Path
import numpy as np
import logging
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from training.data_loader import prepare_datasets
from training.trainer import DistilBERTTrainer, freeze_layers

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PSOOptimizer:
    """PSO-based hyperparameter optimizer for DistilBERT."""
    
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
        self.particle_eval_count = 0  # Total particle evaluations
        
        # Real-time tracking
        self.history = []
        self.animation_data = []
        self.particle_count = 0
        self.log_dir = log_dir or Path(__file__).parent.parent / "logs"
        self.log_dir.mkdir(exist_ok=True)
        
        # Hyperparameter bounds (EXPANDED for better exploration)
        # [learning_rate, batch_size, dropout, frozen_layers]
        self.lb = [1e-6, 4, 0.0, 0]    # Lower bounds - wider range
        self.ub = [1e-4, 64, 0.5, 6]   # Upper bounds - more options
    
    def objective_function(self, params):
        """
        Objective function for PSO to minimize (negative F1 score).
        
        Args:
            params: Array of [learning_rate, batch_size, dropout, frozen_layers]
        
        Returns:
            Negative F1 score (PSO minimizes, we want to maximize F1)
        """
        self.particle_eval_count += 1
        
        # Parse parameters
        learning_rate = params[0]
        batch_size = int(params[1])
        dropout = params[2]
        frozen_layers = int(params[3])
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating Particle {self.particle_eval_count}")
        logger.info(f"{'='*60}")
        logger.info(f"Learning Rate: {learning_rate:.6f}")
        logger.info(f"Batch Size: {batch_size}")
        logger.info(f"Dropout: {dropout:.3f}")
        logger.info(f"Frozen Layers: {frozen_layers}")
        logger.info("⏳ Loading model and preparing training...")
        
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
            
            # Create trainer
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
            
            # Save particle data for animation
            particle_data = {
                'evaluation': self.particle_eval_count,
                'iteration': self.current_iteration if hasattr(self, 'current_iteration') else 0,
                'particle_id': self.particle_count,
                'learning_rate': float(learning_rate),
                'batch_size': int(batch_size),
                'dropout': float(dropout),
                'frozen_layers': int(frozen_layers),
                'f1_score': float(f1_score),
                'accuracy': float(metrics['accuracy']),
                'is_best': bool(is_best)
            }
            self.animation_data.append(particle_data)
            self.particle_count += 1
            
            # Save to file for real-time access
            self._save_progress()
            
            # Return negative F1 (PSO minimizes)
            return -f1_score
        
        except Exception as e:
            logger.error(f"Error in particle evaluation {self.particle_eval_count}: {e}")
            return 0  # Return 0 (worst score) on error
    
    def _save_progress(self):
        """Save current progress to JSON files for real-time access."""
        try:
            # Convert numpy types to Python native types
            def convert_to_native(obj):
                """Convert numpy types to Python native types."""
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
            
            # Save animation data to organized folder
            pso_results_dir = self.log_dir / "results" / "pso"
            pso_results_dir.mkdir(parents=True, exist_ok=True)
            animation_file = pso_results_dir / "animation.json"
            with open(animation_file, 'w') as f:
                json.dump({'data': convert_to_native(self.animation_data)}, f, indent=2)
            
            # Save history data (only save when best improves)
            if self.best_params:
                history_entry = {
                    'evaluation': int(self.particle_eval_count),
                    'best_score': float(self.best_score),
                    'best_params': convert_to_native(self.best_params)
                }
                self.history.append(history_entry)
                
                pso_results_dir = self.log_dir / "results" / "pso"
                pso_results_dir.mkdir(parents=True, exist_ok=True)
                history_file = pso_results_dir / "history.json"
                with open(history_file, 'w') as f:
                    json.dump({'data': convert_to_native(self.history)}, f, indent=2)
            
            # Save current status
            status = {
                'status': 'running',
                'particle_evaluations': int(self.particle_eval_count),
                'best_score': float(self.best_score) if self.best_score > 0 else None,
                'best_params': convert_to_native(self.best_params) if self.best_params else None,
                'total_particles': int(self.particle_count)
            }
            # Save to organized folder
            pso_results_dir = self.log_dir / "results" / "pso"
            pso_results_dir.mkdir(parents=True, exist_ok=True)
            status_file = pso_results_dir / "status.json"
            with open(status_file, 'w') as f:
                json.dump(status, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving progress: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def optimize(self, swarmsize=5, maxiter=10):
        """
        Run PSO optimization.
        
        Args:
            swarmsize: Number of particles in the swarm
            maxiter: Maximum number of iterations
        
        Returns:
            Best hyperparameters found
        """
        import time
        start_time = time.time()
        
        logger.info("\n" + "="*60)
        logger.info("Starting PSO Hyperparameter Optimization")
        logger.info("="*60)
        logger.info(f"Swarm Size: {swarmsize}")
        logger.info(f"Max Iterations: {maxiter}")
        logger.info(f"Search Space:")
        logger.info(f"  Learning Rate: [{self.lb[0]:.6f}, {self.ub[0]:.6f}]")
        logger.info(f"  Batch Size: [{int(self.lb[1])}, {int(self.ub[1])}]")
        logger.info(f"  Dropout: [{self.lb[2]:.2f}, {self.ub[2]:.2f}]")
        logger.info(f"  Frozen Layers: [{int(self.lb[3])}, {int(self.ub[3])}]")
        logger.info(f"\n🚀 Starting PSO with {swarmsize * maxiter} total particle evaluations...")
        logger.info(f"⏳ This will take approximately {(swarmsize * maxiter * 1.5) / 60:.1f} minutes\n")
        sys.stdout.flush()
        sys.stderr.flush()
        
        logger.info("🔄 Initializing PSO swarm...")
        sys.stdout.flush()
        
        # Reset status to starting state
        pso_results_dir = self.log_dir / "results" / "pso"
        pso_results_dir.mkdir(parents=True, exist_ok=True)
        initial_status = {
            'status': 'running',
            'progress': 0,
            'iteration': 0,
            'current_iteration': 0,
            'total_iterations': maxiter,
            'particle': 0,
            'total_particles': swarmsize,
            'message': '🔄 Initializing PSO swarm...'
        }
        with open(pso_results_dir / 'status.json', 'w') as f:
            json.dump(initial_status, f, indent=2)
        logger.info("✅ Status reset to starting state")
        sys.stdout.flush()
        
        # Test: Call objective function once manually to verify it works
        logger.info("🧪 Testing objective function with initial parameters...")
        test_params = [(self.lb[i] + self.ub[i]) / 2 for i in range(len(self.lb))]
        test_result = self.objective_function(test_params)
        logger.info(f"✅ Test complete! Result: {test_result}")
        sys.stdout.flush()
        
        # Run PSO manually (pyswarm hangs with CUDA)
        logger.info("🚀 Starting PSO algorithm...")
        sys.stdout.flush()
        
        # Simple PSO implementation
        n_particles = swarmsize
        n_dims = len(self.lb)
        
        # Initialize particles randomly
        particles = np.random.uniform(self.lb, self.ub, (n_particles, n_dims))
        velocities = np.random.uniform(-0.1, 0.1, (n_particles, n_dims))
        
        # PSO parameters
        w = 0.7  # Inertia weight
        c1 = 1.5  # Cognitive parameter
        c2 = 1.5  # Social parameter
        
        # Track personal best for each particle
        personal_best_positions = particles.copy()
        personal_best_scores = np.full(n_particles, -np.inf)
        
        # Track global best
        global_best_position = particles[0].copy()
        global_best_score = -np.inf
        
        # Evaluate all particles across iterations
        total_evals = 0
        for iteration in range(maxiter):
            # Set current iteration for animation data
            self.current_iteration = iteration
            
            logger.info(f"\n{'='*60}")
            logger.info(f"PSO Iteration {iteration + 1}/{maxiter}")
            logger.info(f"{'='*60}")
            sys.stdout.flush()
            
            for i in range(n_particles):
                total_evals += 1
                
                # Update status BEFORE evaluation starts
                progress = (total_evals / (n_particles * maxiter)) * 100
                status_data = {
                    'status': 'running',
                    'progress': progress,
                    'iteration': iteration + 1,
                    'current_iteration': iteration + 1,
                    'total_iterations': maxiter,
                    'particle': i + 1,
                    'total_particles': n_particles,
                    'message': f'🔄 Training particle {i+1}/{n_particles} (Iteration {iteration+1}/{maxiter})...'
                }
                # Write to correct location for frontend
                pso_results_dir = self.log_dir / "results" / "pso"
                pso_results_dir.mkdir(parents=True, exist_ok=True)
                with open(pso_results_dir / 'status.json', 'w') as f:
                    json.dump(status_data, f, indent=2)
                
                logger.info(f"\n🔄 Starting evaluation of particle {i+1}/{n_particles} in iteration {iteration+1}")
                sys.stdout.flush()
                
                score = -self.objective_function(particles[i])  # Negate because we minimize
                
                logger.info(f"✅ Particle {i+1} evaluation complete! Score: {score:.4f}")
                sys.stdout.flush()
                
                # Get detailed metrics from last evaluation
                accuracy = self.best_params.get('accuracy', 0) if self.best_params else 0
                
                # Save particle data for visualization with detailed metrics
                particle_data = {
                    'iteration': iteration + 1,
                    'particle_id': i + 1,
                    'position': particles[i].tolist(),
                    'score': float(score),
                    'f1_score': float(score),
                    'accuracy': float(accuracy),
                    'params': {
                        'learning_rate': float(particles[i][0]),
                        'batch_size': int(particles[i][1]),
                        'dropout': float(particles[i][2]),
                        'frozen_layers': int(particles[i][3])
                    },
                    'timestamp': __import__('datetime').datetime.now().isoformat()
                }
                self.animation_data.append(particle_data)
                
                # Update status with current particle metrics for frontend
                status_data['current_particle_metrics'] = {
                    'particle_id': i + 1,
                    'f1_score': float(score),
                    'accuracy': float(accuracy),
                    'learning_rate': float(particles[i][0]),
                    'batch_size': int(particles[i][1]),
                    'dropout': float(particles[i][2]),
                    'frozen_layers': int(particles[i][3])
                }
                status_data['best_so_far'] = {
                    'f1_score': float(self.best_score),
                    'params': self.best_params if self.best_params else {}
                }
                
                # Write updated status with metrics
                with open(pso_results_dir / 'status.json', 'w') as f:
                    json.dump(status_data, f, indent=2)
                
                # Update personal best
                if score > personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = particles[i].copy()
                
                # Update global best
                if score > global_best_score:
                    global_best_score = score
                    global_best_position = particles[i].copy()
                    self.best_score = score
                    logger.info(f"🎉 New global best F1: {self.best_score:.4f}")
                    
                    # Save to history
                    history_entry = {
                        'iteration': iteration + 1,
                        'particle': i + 1,
                        'evaluation': total_evals,
                        'best_score': float(self.best_score),
                        'best_params': {
                            'learning_rate': float(global_best_position[0]),
                            'batch_size': int(global_best_position[1]),
                            'dropout': float(global_best_position[2]),
                            'frozen_layers': int(global_best_position[3])
                        }
                    }
                    self.history.append(history_entry)
                    
                    # Save history to file
                    pso_results_dir = self.log_dir / "results" / "pso"
                    with open(pso_results_dir / 'history.json', 'w') as f:
                        json.dump({'data': self.history}, f, indent=2)
                    
                    sys.stdout.flush()
            
            # Update velocities and positions for next iteration (PSO update equations)
            for i in range(n_particles):
                r1 = np.random.random(n_dims)
                r2 = np.random.random(n_dims)
                
                # Velocity update: v = w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x)
                velocities[i] = (w * velocities[i] + 
                                c1 * r1 * (personal_best_positions[i] - particles[i]) +
                                c2 * r2 * (global_best_position - particles[i]))
                
                # Position update: x = x + v
                particles[i] = particles[i] + velocities[i]
                
                # Enforce bounds
                particles[i] = np.clip(particles[i], self.lb, self.ub)
        
        logger.info(f"\n✅ PSO Complete! Best F1: {self.best_score:.4f}")
        sys.stdout.flush()
        
        # Ensure best_params is populated FIRST
        logger.info(f"Checking best_params: {self.best_params}")
        sys.stdout.flush()
        if not self.best_params or self.best_params.get('f1_score', 0) < self.best_score:
            logger.info("Creating best_params from global_best_position")
            self.best_params = {
                'learning_rate': float(global_best_position[0]),
                'batch_size': int(global_best_position[1]),
                'dropout': float(global_best_position[2]),
                'frozen_layers': int(global_best_position[3]),
                'f1_score': float(self.best_score),
                'accuracy': 0.0  # Will be updated from objective function
            }
        
        logger.info(f"Final best_params: {self.best_params}")
        sys.stdout.flush()
        
        # Calculate total optimization time FIRST
        end_time = time.time()
        optimization_time = end_time - start_time
        
        # IMMEDIATELY save latest.json BEFORE status
        pso_results_dir = self.log_dir / "results" / "pso"
        pso_results_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            logger.info(f"💾 SAVING latest.json NOW...")
            sys.stdout.flush()
            
            # Convert numpy types helper
            def convert_to_native(obj):
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
            
            latest_result = {
                'type': 'pso',
                'completedAt': datetime.now().isoformat(),
                'accuracy': float(self.best_params.get('accuracy', 0)) if self.best_params else 0.0,
                'f1_score': float(self.best_score),
                'message': f'PSO Optimization completed - Best F1: {self.best_score:.4f}',
                'best_params': convert_to_native(self.best_params) if self.best_params else {},
                'optimization_params': {
                    'swarmsize': swarmsize,
                    'maxiter': maxiter
                },
                'total_iterations': maxiter,
                'best_iteration': int(self.particle_eval_count),
                'training_time': float(optimization_time)
            }
            latest_file = pso_results_dir / "latest.json"
            with open(latest_file, 'w') as f:
                json.dump(latest_result, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            logger.info(f"✅ SAVED latest.json!")
            sys.stdout.flush()
        except Exception as e:
            import traceback
            logger.error(f"❌ Error saving latest.json: {e}")
            logger.error(traceback.format_exc())
            sys.stdout.flush()
        
        # NOW update status to completed
        try:
            completion_status = {
                'status': 'completed',
                'progress': 100,
                'iteration': maxiter,
                'current_iteration': maxiter,
                'total_iterations': maxiter,
                'message': '✅ PSO optimization completed successfully!',
                'best_score': float(self.best_score),
                'best_params': self.best_params
            }
            with open(pso_results_dir / 'status.json', 'w') as f:
                json.dump(completion_status, f, indent=2)
            logger.info("✅ Status set to completed")
            sys.stdout.flush()
        except Exception as e:
            logger.error(f"Error saving status: {e}")
            import traceback
            logger.error(traceback.format_exc())
            sys.stdout.flush()
        
        # Log final results
        logger.info("\n" + "="*60)
        logger.info("PSO Optimization Complete!")
        logger.info("="*60)
        logger.info(f"Best F1 Score: {self.best_score:.4f}")
        if self.best_params:
            logger.info(f"  Learning Rate: {self.best_params.get('learning_rate', 'N/A')}")
            logger.info(f"  Batch Size: {self.best_params.get('batch_size', 'N/A')}")
            logger.info(f"  Dropout: {self.best_params.get('dropout', 'N/A')}")
            logger.info(f"  Frozen Layers: {self.best_params.get('frozen_layers', 'N/A')}")
        sys.stdout.flush()
        
        return self.best_params


def run_pso_optimization():
    """Main function to run PSO optimization."""
    
    # Get parameters from command line or use defaults
    import sys
    SWARMSIZE = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    MAXITER = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    
    # Configuration
    MODEL_NAME = "distilbert-base-uncased"
    MAX_LENGTH = 128
    NUM_EPOCHS = 1  # 1 epoch per particle
    
    # Paths
    BASE_DIR = Path(__file__).parent.parent
    MODELS_DIR = BASE_DIR / "models"
    MODELS_DIR.mkdir(exist_ok=True)
    
    BEST_PARAMS_PATH = MODELS_DIR / "best_hyperparameters.json"
    
    # Load tokenizer
    logger.info(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
    
    # Prepare datasets - train on VERY small subset, test on full data for realistic accuracy gap
    logger.info("Preparing datasets...")
    SUBSET_FRACTION = 0.05  # Use only 5% of data for training to show accuracy gap
    
    # Get full datasets first
    train_dataset_full, val_dataset_full, test_dataset_full = prepare_datasets(
        tokenizer, 
        MAX_LENGTH,
        subset_fraction=1.0  # Full data for validation/testing
    )
    
    # Get small subset for training only
    train_dataset_subset, _, _ = prepare_datasets(
        tokenizer, 
        MAX_LENGTH,
        subset_fraction=SUBSET_FRACTION
    )
    
    logger.info(f"⚠️  Training on only {SUBSET_FRACTION*100:.0f}% of data, testing on 100% to demonstrate generalization gap")
    logger.info(f"Train size: {len(train_dataset_subset)}, Val size: {len(val_dataset_full)}")
    
    # Initialize optimizer
    logger.info("Initializing PSO optimizer...")
    optimizer = PSOOptimizer(
        train_dataset=train_dataset_subset,
        val_dataset=val_dataset_full,
        num_epochs=NUM_EPOCHS
    )
    
    # Run optimization
    best_params = optimizer.optimize(swarmsize=SWARMSIZE, maxiter=MAXITER)
    
    # Save best hyperparameters
    with open(BEST_PARAMS_PATH, 'w') as f:
        json.dump(best_params, f, indent=2)
    
    logger.info(f"Best hyperparameters saved to {BEST_PARAMS_PATH}")
    logger.info("Optimization complete!")


if __name__ == "__main__":
    run_pso_optimization()
