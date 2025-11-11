"""Train baseline DistilBERT model with custom hyperparameters."""
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import json
import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from training.data_loader import prepare_datasets
from training.trainer import DistilBERTTrainer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_baseline(learning_rate=2e-5, batch_size=16, dropout=0.1, frozen_layers=0, epochs=3):
    """Train baseline model with custom hyperparameters."""
    
    # Configuration
    MODEL_NAME = "distilbert-base-uncased"
    NUM_LABELS = 6
    MAX_LENGTH = 128
    
    # Use provided hyperparameters
    LEARNING_RATE = learning_rate
    BATCH_SIZE = batch_size
    NUM_EPOCHS = epochs
    DROPOUT = dropout
    FROZEN_LAYERS = frozen_layers
    
    # Paths
    BASE_DIR = Path(__file__).parent.parent
    MODELS_DIR = BASE_DIR / "models"
    MODELS_DIR.mkdir(exist_ok=True)
    
    MODEL_PATH = MODELS_DIR / "baseline_model.pt"
    METRICS_PATH = MODELS_DIR / "baseline_metrics.json"
    
    logger.info("="*50)
    logger.info("Training Baseline DistilBERT Model")
    logger.info("="*50)
    
    # Load tokenizer
    logger.info(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
    
    # Prepare datasets
    logger.info("Preparing datasets...")
    train_dataset, val_dataset, test_dataset = prepare_datasets(tokenizer, MAX_LENGTH)
    
    # Initialize model
    logger.info("Initializing model...")
    from transformers import DistilBertConfig
    config = DistilBertConfig.from_pretrained(MODEL_NAME)
    config.num_labels = NUM_LABELS
    config.dropout = DROPOUT
    config.attention_dropout = DROPOUT
    config.seq_classif_dropout = DROPOUT
    
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        config=config
    )
    
    # Freeze layers if specified
    if FROZEN_LAYERS > 0:
        logger.info(f"Freezing first {FROZEN_LAYERS} transformer layers")
        for i, layer in enumerate(model.distilbert.transformer.layer):
            if i < FROZEN_LAYERS:
                for param in layer.parameters():
                    param.requires_grad = False
    
    # Create trainer
    trainer = DistilBERTTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
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
    
    logger.info("\n" + "="*50)
    logger.info("Baseline Training Complete!")
    logger.info("="*50)
    logger.info(f"Final Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Final F1 Score: {metrics['f1_weighted']:.4f}")
    
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train baseline DistilBERT model')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--frozen_layers', type=int, default=0, help='Number of frozen layers')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    
    args = parser.parse_args()
    
    train_baseline(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        dropout=args.dropout,
        frozen_layers=args.frozen_layers,
        epochs=args.epochs
    )
