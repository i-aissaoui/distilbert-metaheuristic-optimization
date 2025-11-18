"""Genetic Algorithm-based hyperparameter optimization for DistilBERT."""
import os
import warnings
warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import json
import sys
from pathlib import Path
import numpy as np
import logging
import random
from datetime import datetime
from deap import base, creator, tools, algorithms

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from training.data_loader import prepare_datasets
from training.trainer import DistilBERTTrainer, freeze_layers

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GAOptimizer:
    """Genetic Algorithm-based hyperparameter optimizer for DistilBERT."""
    
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
        self.generation = 0
        # Track last evaluated individual's metrics (primary latest values)
        self.last_metrics = {"f1": 0.0, "accuracy": 0.0}
        
        # Real-time tracking
        self.history = []
        self.animation_data = []
        self.individual_count = 0
        self.log_dir = log_dir or Path(__file__).parent.parent / "logs"
        self.log_dir.mkdir(exist_ok=True)
        
        # Hyperparameter bounds (EXPANDED for better exploration)
        # [learning_rate, batch_size, dropout, frozen_layers]
        self.bounds = {
            'learning_rate': (1e-6, 1e-4),
            'batch_size': (4, 64),
            'dropout': (0.0, 0.5),
            'frozen_layers': (0, 6)
        }
        
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

        # Setup DEAP
        self._setup_deap()
    
    def _setup_deap(self):
        """Setup DEAP genetic algorithm framework."""
        # Create fitness and individual classes
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMax)
        
        self.toolbox = base.Toolbox()
        
        # Attribute generators
        self.toolbox.register("learning_rate", random.uniform, 
                             self.bounds['learning_rate'][0], 
                             self.bounds['learning_rate'][1])
        self.toolbox.register("batch_size", random.randint, 
                             self.bounds['batch_size'][0], 
                             self.bounds['batch_size'][1])
        self.toolbox.register("dropout", random.uniform, 
                             self.bounds['dropout'][0], 
                             self.bounds['dropout'][1])
        self.toolbox.register("frozen_layers", random.randint, 
                             self.bounds['frozen_layers'][0], 
                             self.bounds['frozen_layers'][1])
        
        # Structure initializers
        self.toolbox.register("individual", tools.initCycle, creator.Individual,
                             (self.toolbox.learning_rate, self.toolbox.batch_size,
                              self.toolbox.dropout, self.toolbox.frozen_layers), n=1)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        
        # Genetic operators
        self.toolbox.register("evaluate", self.evaluate_individual)
        self.toolbox.register("mate", tools.cxBlend, alpha=0.5)
        self.toolbox.register("mutate", self.mutate_individual)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
    
    def evaluate_individual(self, individual):
        """
        Evaluate fitness of an individual (hyperparameter configuration).
        
        Args:
            individual: List of [learning_rate, batch_size, dropout, frozen_layers]
        
        Returns:
            Tuple with F1 score
        """
        self.individual_count += 1
        
        # Parse parameters with optimize/fixed handling
        # First, clamp raw genes to bounds to avoid invalid values from crossover
        lr_gene = float(individual[0])
        bs_gene = int(round(individual[1]))
        dr_gene = float(individual[2])
        fr_gene = int(round(individual[3]))

        # Clamp to bounds
        lr_min, lr_max = self.bounds['learning_rate']
        bs_min, bs_max = self.bounds['batch_size']
        dr_min, dr_max = self.bounds['dropout']
        fl_min, fl_max = self.bounds['frozen_layers']

        lr_gene = max(lr_min, min(lr_max, lr_gene))
        bs_gene = max(int(bs_min), min(int(bs_max), bs_gene))
        dr_gene = max(dr_min, min(dr_max, dr_gene))
        fr_gene = max(int(fl_min), min(int(fl_max), fr_gene))

        learning_rate = lr_gene if self.optimize_mask.get('learning_rate', True) else float(self.fixed_values['learning_rate'])
        batch_size = max(4, bs_gene) if self.optimize_mask.get('batch_size', True) else int(self.fixed_values['batch_size'])
        dropout = dr_gene if self.optimize_mask.get('dropout', True) else float(self.fixed_values['dropout'])
        frozen_layers = fr_gene if self.optimize_mask.get('frozen_layers', True) else int(self.fixed_values['frozen_layers'])
        
        logger.info(f"\n{'='*60}")
        logger.info(f"GA Generation {self.generation} - Individual {self.individual_count}")
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
            # Validate datasets have proper tokenizer (fix corruption issue)
            if hasattr(self.train_dataset, 'tokenizer'):
                if not callable(self.train_dataset.tokenizer):
                    logger.error(f"Train dataset tokenizer is corrupted! Type: {type(self.train_dataset.tokenizer)}")
                    # Re-load tokenizer
                    from transformers import DistilBertTokenizer
                    self.train_dataset.tokenizer = DistilBertTokenizer.from_pretrained(self.model_name, local_files_only=True)
                    self.val_dataset.tokenizer = DistilBertTokenizer.from_pretrained(self.model_name, local_files_only=True)
                    logger.info("✅ Tokenizer restored")
            
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
            
            # Ensure f1_score is a float
            f1_score = float(metrics.get('f1_weighted', 0))
            accuracy = float(metrics.get('accuracy', 0))

            # Always record last evaluated individual's metrics
            self.last_metrics = {"f1": f1_score, "accuracy": accuracy}
            
            logger.info(f"F1 Score: {f1_score:.4f}")
            logger.info(f"Accuracy: {accuracy:.4f}")
            
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
                    'accuracy': accuracy
                }
                logger.info(f"🎉 New best configuration found!")
            
            # Save individual data for animation
            individual_data = {
                'generation': self.generation,
                'individual_id': self.individual_count,
                'learning_rate': float(learning_rate),
                'batch_size': int(batch_size),
                'dropout': float(dropout),
                'frozen_layers': int(frozen_layers),
                'f1_score': float(f1_score),
                'accuracy': float(accuracy),
                'is_best': is_best
            }
            self.animation_data.append(individual_data)
            
            # Save to file for real-time access
            self._save_progress()
            
            return (f1_score,)
        
        except Exception as e:
            import traceback
            import sys
            
            # Get detailed error info
            exc_type, exc_value, exc_tb = sys.exc_info()
            tb_lines = traceback.format_exception(exc_type, exc_value, exc_tb)
            
            error_msg = f"Error in generation {self.generation}: {e}"
            logger.error(error_msg)
            logger.error(f"Exception type: {exc_type.__name__}")
            logger.error(f"Exception value: {exc_value}")
            
            # Log each line of traceback separately
            for line in tb_lines:
                logger.error(line.rstrip())
            
            return (0.0,)
    
    def mutate_individual(self, individual):
        """
        Mutate an individual with bounds checking and error handling.
        
        Args:
            individual: Individual to mutate
        
        Returns:
            Tuple with mutated individual
        """
        try:
            # Get parameter bounds with safety checks
            lr_min, lr_max = self.bounds.get('learning_rate', (1e-6, 1e-4))
            bs_min, bs_max = self.bounds.get('batch_size', (4, 64))
            dr_min, dr_max = self.bounds.get('dropout', (0.0, 0.5))
            fl_min, fl_max = self.bounds.get('frozen_layers', (0, 6))
            
            # Ensure individual has the correct number of genes
            if len(individual) < 4:
                logger.warning(f"Individual has incorrect number of genes: {len(individual)}")
                return individual,
            
            # Apply mutation with bounds checking and validation
            try:
                if self.optimize_mask.get('learning_rate', True):
                    individual[0] = max(lr_min, min(lr_max, individual[0] * random.uniform(0.8, 1.2)))
                    individual[0] = float(individual[0])  # Ensure it's a float
                
                if self.optimize_mask.get('batch_size', True):
                    new_bs = int(individual[1] * random.uniform(0.8, 1.2))
                    individual[1] = max(bs_min, min(bs_max, new_bs))
                    individual[1] = int(individual[1])  # Ensure it's an int
                
                if self.optimize_mask.get('dropout', True):
                    individual[2] = max(dr_min, min(dr_max, individual[2] * random.uniform(0.8, 1.2)))
                    individual[2] = float(individual[2])  # Ensure it's a float
                
                if self.optimize_mask.get('frozen_layers', True):
                    fl = int(individual[3]) + random.choice([-1, 0, 1])
                    individual[3] = max(fl_min, min(fl_max, fl))
                    individual[3] = int(individual[3])  # Ensure it's an int
                
                logger.debug(f"Mutated individual to: {individual}")
                return individual,
                
            except (ValueError, TypeError) as ve:
                logger.error(f"Value error during mutation: {ve}")
                # Return a new random individual if mutation fails
                return self.toolbox.individual(),
                
        except Exception as e:
            logger.error(f"Unexpected error in mutation: {str(e)}")
            # Return a new random individual if something goes wrong
            return self.toolbox.individual(),
        
        return (individual,)
    
    def _save_progress(self):
        """Save current progress to JSON files for real-time access."""
        try:
            # Save animation data
            animation_file = self.log_dir / "ga_animation.json"
            with open(animation_file, 'w') as f:
                json.dump({'data': self.animation_data}, f, indent=2)
            
            # Save history data
            if self.best_params:
                history_entry = {
                    'generation': self.generation,
                    'best_score': float(self.best_score),
                    'best_params': self.best_params
                }
                
                # Update or append history
                if self.generation < len(self.history):
                    self.history[self.generation] = history_entry
                else:
                    self.history.append(history_entry)
                
                ga_results_dir = self.log_dir / "results" / "ga"
                ga_results_dir.mkdir(parents=True, exist_ok=True)
                history_file = ga_results_dir / "history.json"
                with open(history_file, 'w') as f:
                    json.dump({'data': self.history}, f, indent=2)
            
            # Save current status with progress
            # Calculate progress based on generation (need total_generations from optimize method)
            # We'll update this in the optimize method instead
            pass
                
        except Exception as e:
            logger.error(f"Error saving progress: {e}")
    
    def optimize(self, population_size=5, num_generations=10, crossover_prob=0.7, mutation_prob=0.2):
        """
        Run Genetic Algorithm optimization.
        
        Args:
            population_size: Number of individuals in population
            num_generations: Number of generations to evolve
            crossover_prob: Probability of crossover
            mutation_prob: Probability of mutation
        
        Returns:
            Best hyperparameters found
        """
        import time
        start_time = time.time()
        
        logger.info("\n" + "="*60)
        logger.info("Starting Genetic Algorithm Hyperparameter Optimization")
        logger.info("="*60)
        logger.info(f"Population Size: {population_size}")
        logger.info(f"Num Generations: {num_generations}")
        logger.info(f"Crossover Probability: {crossover_prob}")
        logger.info(f"Mutation Probability: {mutation_prob}")
        logger.info(f"Search Space:")
        logger.info(f"  Learning Rate: {self.bounds['learning_rate']}")
        logger.info(f"  Batch Size: {self.bounds['batch_size']}")
        logger.info(f"  Dropout: {self.bounds['dropout']}")
        logger.info(f"  Frozen Layers: {self.bounds['frozen_layers']}")
        logger.info("Starting Parameter Modes:")
        for name in self.param_names:
            mode_str = "optimized" if self.optimize_mask.get(name, True) else f"fixed ({self.fixed_values.get(name)})"
            logger.info(f"Starting - {name}: {mode_str}")
        
        # Create initial status file IMMEDIATELY
        ga_results_dir = self.log_dir / "results" / "ga"
        ga_results_dir.mkdir(parents=True, exist_ok=True)
        initial_status = {
            'status': 'running',
            'progress': 0,
            'generation': 0,
            'total_generations': num_generations,
            'best_score': 0,
            'best_params': None,
            'message': '🔄 Initializing GA optimization...'
        }
        with open(ga_results_dir / 'status.json', 'w') as f:
            json.dump(initial_status, f, indent=2)
        logger.info("✅ Initial status file created")
        
        # Create initial population
        population = self.toolbox.population(n=population_size)
        
        # Statistics
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        # Run evolution
        for gen in range(num_generations):
            self.generation = gen
            logger.info(f"\n{'#'*60}")
            logger.info(f"Generation {gen + 1}/{num_generations}")
            logger.info(f"{'#'*60}")
            
            # Evaluate the entire population with better error handling
            try:
                fitnesses = []
                for ind in population:
                    try:
                        fit = self.toolbox.evaluate(ind)
                        fitnesses.append(fit)
                    except Exception as e:
                        logger.error(f"Error evaluating individual {ind}: {str(e)}")
                        # Assign a very low fitness to failed evaluations
                        fitnesses.append((0.0,))
                
                # Assign fitnesses to individuals
                for ind, fit in zip(population, fitnesses):
                    ind.fitness.values = fit
                    
            except Exception as e:
                logger.error(f"Critical error in population evaluation: {str(e)}")
                # Assign minimum fitness to all individuals to continue
                for ind in population:
                    ind.fitness.values = (0.0,)
                    
            # Log statistics
            record = stats.compile(population)
            logger.info(f"Generation {gen + 1} Stats:")
            logger.info(f"  Max F1: {record['max']:.4f}")
            logger.info(f"  Avg F1: {record['avg']:.4f}")
            logger.info(f"  Min F1: {record['min']:.4f}")
            logger.info(f"  Std F1: {record['std']:.4f}")
            
            # Update status with progress
            progress = ((gen + 1) / num_generations) * 100
            ga_results_dir = self.log_dir / "results" / "ga"
            ga_results_dir.mkdir(parents=True, exist_ok=True)
            status = {
                'status': 'running',
                'progress': float(progress),
                'generation': gen + 1,
                'total_generations': num_generations,
                'best_score': float(self.best_score) if self.best_score > 0 else 0,
                'best_params': self.best_params,
                'message': f'🔄 Generation {gen + 1}/{num_generations} - Best F1: {self.best_score:.4f}'
            }
            with open(ga_results_dir / 'status.json', 'w') as f:
                json.dump(status, f, indent=2)
            
            # Select next generation
            offspring = self.toolbox.select(population, len(population))
            offspring = list(map(self.toolbox.clone, offspring))
            
            # Apply crossover
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < crossover_prob:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            # Enforce bounds after crossover
            for ind in offspring:
                try:
                    # learning_rate
                    ind[0] = float(max(self.bounds['learning_rate'][0], min(self.bounds['learning_rate'][1], ind[0])))
                    # batch_size (int)
                    ind[1] = int(max(self.bounds['batch_size'][0], min(self.bounds['batch_size'][1], round(ind[1]))))
                    # dropout
                    ind[2] = float(max(self.bounds['dropout'][0], min(self.bounds['dropout'][1], ind[2])))
                    # frozen_layers (int)
                    ind[3] = int(max(self.bounds['frozen_layers'][0], min(self.bounds['frozen_layers'][1], round(ind[3]))))
                except Exception:
                    # If anything goes wrong, regenerate a valid individual
                    new_ind = self.toolbox.individual()
                    ind[:] = new_ind[:]

            # Apply mutation
            for mutant in offspring:
                if random.random() < mutation_prob:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Final bounds enforcement after mutation
            for ind in offspring:
                try:
                    ind[0] = float(max(self.bounds['learning_rate'][0], min(self.bounds['learning_rate'][1], ind[0])))
                    ind[1] = int(max(self.bounds['batch_size'][0], min(self.bounds['batch_size'][1], round(ind[1]))))
                    ind[2] = float(max(self.bounds['dropout'][0], min(self.bounds['dropout'][1], ind[2])))
                    ind[3] = int(max(self.bounds['frozen_layers'][0], min(self.bounds['frozen_layers'][1], round(ind[3]))))
                except Exception:
                    new_ind = self.toolbox.individual()
                    ind[:] = new_ind[:]
            
            # Replace population
            population[:] = offspring
        
        # IMMEDIATELY update status to completed after loop
        ga_results_dir = self.log_dir / "results" / "ga"
        ga_results_dir.mkdir(parents=True, exist_ok=True)
        completion_status = {
            'status': 'completed',
            'progress': 100,
            'generation': num_generations,
            'message': '✅ GA optimization completed successfully!',
            'best_score': float(self.best_score),
            'best_accuracy': float(self.best_params.get('accuracy', 0)) if self.best_params else 0.0
        }
        with open(ga_results_dir / 'status.json', 'w') as f:
            json.dump(completion_status, f, indent=2)
        logger.info("✅ Status immediately set to completed")
        
        # Calculate total optimization time
        end_time = time.time()
        optimization_time = end_time - start_time
        
        logger.info("\n" + "="*60)
        logger.info("Genetic Algorithm Optimization Complete!")
        logger.info("="*60)
        logger.info(f"Best F1 Score: {self.best_score:.4f}")
        logger.info(f"Best Accuracy: {float(self.best_params.get('accuracy', 0)):.4f}")
        logger.info(f"⏱️  Total optimization time: {optimization_time/60:.2f} minutes ({optimization_time:.1f} seconds)")
        logger.info(f"Best Configuration:")
        for key, value in self.best_params.items():
            logger.info(f"  {key}: {value}")
        # Explicit single-line result for terminal filters
        logger.info(
            f"GA RESULT -> F1: {self.best_score:.4f}, Acc: {float(self.best_params.get('accuracy', 0)):.4f}, "
            f"Params: lr={self.best_params.get('learning_rate')}, bs={self.best_params.get('batch_size')}, "
            f"dropout={self.best_params.get('dropout')}, frozen={self.best_params.get('frozen_layers')}"
        )
        
        # Save final status
        final_status = {
            'status': 'completed',
            'generation': self.generation,
            'best_score': float(self.best_score),
            'best_params': self.best_params,
            'total_individuals': self.individual_count,
            'total_generations': num_generations
        }
        ga_results_dir = self.log_dir / "results" / "ga"
        ga_results_dir.mkdir(parents=True, exist_ok=True)
        status_file = ga_results_dir / "status.json"
        with open(status_file, 'w') as f:
            json.dump(final_status, f, indent=2)
        
        # Save to latest.json
        latest_result = {
            'type': 'ga',
            'completedAt': datetime.now().isoformat(),
            'accuracy': float(self.last_metrics.get('accuracy', 0.0)),
            'f1_score': float(self.last_metrics.get('f1', 0.0)),
            'message': f'GA Optimization completed - Best F1: {self.best_score:.4f}',
            'best_params': self.best_params,
            'optimization_params': {
                'population_size': population_size,
                'num_generations': num_generations
            },
            'training_time': float(optimization_time),  # in seconds
            'parameter_selection': {
                'optimize': self.optimize_mask,
                'fixed': self.fixed_values
            },
            # Placeholders indicating 100% full-data training not yet performed
            'full_data_f1_score': 0.0,
            'full_data_accuracy': 0.0,
            # Also include best-of-run for reference
            'best_score': float(self.best_score)
        }
        latest_file = ga_results_dir / "latest.json"
        with open(latest_file, 'w') as f:
            json.dump(latest_result, f, indent=2)
        logger.info(f"✅ Saved GA results to latest.json")
        # Final explicit saved line
        logger.info(
            f"GA SAVED -> latest.json | Best F1: {self.best_score:.4f}, Acc: {float(self.best_params.get('accuracy', 0)):.4f}"
        )
        
        return self.best_params


def run_ga_optimization():
    """Main function to run GA optimization."""
    
    # Get parameters from command line or use defaults
    import sys
    POPULATION_SIZE = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    NUM_GENERATIONS = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    
    # Configuration
    MODEL_NAME = "distilbert-base-uncased"
    MAX_LENGTH = 128
    NUM_EPOCHS = 1  # 1 epoch to prevent overfitting on small data
    
    logger.info("Starting optimization...")
    logger.info(f"Starting Genetic Algorithm Hyperparameter Optimization")
    logger.info(f"Population Size: {POPULATION_SIZE}")
    logger.info(f"Num Generations: {NUM_GENERATIONS}")
    
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
    
    logger.info("Initializing GA optimizer...")
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

    optimizer = GAOptimizer(
        train_dataset=train_dataset_subset,
        val_dataset=val_dataset_full,
        num_epochs=NUM_EPOCHS,
        optimize_mask=optimize,
        fixed_values=fixed
    )
    
    logger.info("Starting optimization...")
    best_params = optimizer.optimize(
        population_size=POPULATION_SIZE,
        num_generations=NUM_GENERATIONS,
        crossover_prob=0.7,
        mutation_prob=0.2
    )
    
    # Save best hyperparameters
    output_file = Path(__file__).parent.parent / "models" / "ga_best_hyperparameters.json"
    output_file.parent.mkdir(exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(best_params, f, indent=2)
    
    logger.info(f"\nBest hyperparameters saved to: {output_file}")
    logger.info("Optimization complete!")


if __name__ == "__main__":
    run_ga_optimization()
