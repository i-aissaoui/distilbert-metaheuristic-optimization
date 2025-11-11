"""Train optimized DistilBERT model using PSO-found hyperparameters."""
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from training.data_loader import prepare_datasets
from training.trainer import DistilBERTTrainer, freeze_layers
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_optimized():
    """Train optimized model using PSO-found hyperparameters."""
    
    # Configuration
    MODEL_NAME = "distilbert-base-uncased"
    NUM_LABELS = 6
    MAX_LENGTH = 128
    NUM_EPOCHS = 3  # Full training epochs
    
    # Paths
    BASE_DIR = Path(__file__).parent.parent
    MODELS_DIR = BASE_DIR / "models"
    MODELS_DIR.mkdir(exist_ok=True)
    
    BEST_PARAMS_PATH = MODELS_DIR / "best_hyperparameters.json"
    MODEL_PATH = MODELS_DIR / "optimized_model.pt"
    METRICS_PATH = MODELS_DIR / "optimized_metrics.json"
    COMPARISON_PATH = MODELS_DIR / "performance_comparison.json"
    
    # Load best hyperparameters
    if not BEST_PARAMS_PATH.exists():
        logger.error(f"Best hyperparameters not found at {BEST_PARAMS_PATH}")
        logger.error("Please run pso_optimization.py first!")
        return
    
    logger.info(f"Loading best hyperparameters from {BEST_PARAMS_PATH}")
    with open(BEST_PARAMS_PATH, 'r') as f:
        best_params = json.load(f)
    
    logger.info("="*50)
    logger.info("Training Optimized DistilBERT Model")
    logger.info("="*50)
    logger.info("Using PSO-optimized hyperparameters:")
    for key, value in best_params.items():
        logger.info(f"  {key}: {value}")
    
    # Load tokenizer
    logger.info(f"\nLoading tokenizer: {MODEL_NAME}")
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
    
    # Prepare datasets
    logger.info("Preparing datasets...")
    train_dataset, val_dataset, test_dataset = prepare_datasets(tokenizer, MAX_LENGTH)
    
    # Initialize model with optimized hyperparameters
    logger.info("Initializing model...")
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        hidden_dropout_prob=best_params['dropout'],
        attention_probs_dropout_prob=best_params['dropout']
    )
    
    # Freeze layers if specified
    if best_params['frozen_layers'] > 0:
        freeze_layers(model, best_params['frozen_layers'])
    
    # Create trainer
    trainer = DistilBERTTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        learning_rate=best_params['learning_rate'],
        batch_size=int(best_params['batch_size']),
        num_epochs=NUM_EPOCHS
    )
    
    # Train
    logger.info("\nStarting training...")
    metrics = trainer.train()
    
    # Save model
    logger.info(f"\nSaving model to {MODEL_PATH}")
    torch.save(model.state_dict(), MODEL_PATH)
    
    # Save metrics
    logger.info(f"Saving metrics to {METRICS_PATH}")
    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Create performance comparison
    baseline_metrics_path = MODELS_DIR / "baseline_metrics.json"
    if baseline_metrics_path.exists():
        with open(baseline_metrics_path, 'r') as f:
            baseline_metrics = json.load(f)
        
        comparison = {
            "baseline_metrics": baseline_metrics,
            "optimized_metrics": metrics,
            "improvement": {
                "accuracy": metrics['accuracy'] - baseline_metrics['accuracy'],
                "f1_weighted": metrics['f1_weighted'] - baseline_metrics['f1_weighted'],
                "f1_macro": metrics['f1_macro'] - baseline_metrics['f1_macro']
            },
            "improvement_percentage": {
                "accuracy": ((metrics['accuracy'] - baseline_metrics['accuracy']) / baseline_metrics['accuracy'] * 100),
                "f1_weighted": ((metrics['f1_weighted'] - baseline_metrics['f1_weighted']) / baseline_metrics['f1_weighted'] * 100),
                "f1_macro": ((metrics['f1_macro'] - baseline_metrics['f1_macro']) / baseline_metrics['f1_macro'] * 100)
            }
        }
        
        with open(COMPARISON_PATH, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        logger.info("\n" + "="*50)
        logger.info("Performance Comparison")
        logger.info("="*50)
        logger.info(f"Baseline Accuracy: {baseline_metrics['accuracy']:.4f}")
        logger.info(f"Optimized Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Improvement: {comparison['improvement']['accuracy']:.4f} ({comparison['improvement_percentage']['accuracy']:.2f}%)")
        logger.info(f"\nBaseline F1: {baseline_metrics['f1_weighted']:.4f}")
        logger.info(f"Optimized F1: {metrics['f1_weighted']:.4f}")
        logger.info(f"Improvement: {comparison['improvement']['f1_weighted']:.4f} ({comparison['improvement_percentage']['f1_weighted']:.2f}%)")
    
    logger.info("\n" + "="*50)
    logger.info("Optimized Training Complete!")
    logger.info("="*50)
    logger.info(f"Final Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Final F1 Score: {metrics['f1_weighted']:.4f}")
    
    return metrics


if __name__ == "__main__":
    train_optimized()
