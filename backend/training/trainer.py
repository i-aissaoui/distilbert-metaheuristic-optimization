"""Training utilities for DistilBERT model."""
import os
import warnings
warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

import torch
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from tqdm import tqdm
import logging
from typing import Dict, Tuple
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DistilBERTTrainer:
    """Trainer class for DistilBERT model."""
    
    def __init__(
        self,
        model,
        train_dataset,
        val_dataset,
        learning_rate: float = 2e-5,
        batch_size: int = 16,
        num_epochs: int = 3,
        warmup_steps: int = 0,
        device: str = None
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.warmup_steps = warmup_steps
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model.to(self.device)
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False
        )
        
        # Setup optimizer and scheduler
        self.optimizer = AdamW(model.parameters(), lr=learning_rate)
        total_steps = len(self.train_loader) * num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
    
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    def evaluate(self) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
        """Evaluate model on validation set."""
        self.model.eval()
        all_preds = []
        all_labels = []
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                
                logits = outputs.logits
                preds = torch.argmax(logits, dim=-1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        metrics = {
            'loss': total_loss / len(self.val_loader),
            'accuracy': accuracy_score(all_labels, all_preds),
            'f1_weighted': f1_score(all_labels, all_preds, average='weighted'),
            'f1_macro': f1_score(all_labels, all_preds, average='macro'),
            'precision': precision_score(all_labels, all_preds, average='weighted'),
            'recall': recall_score(all_labels, all_preds, average='weighted')
        }
        
        return metrics, all_preds, all_labels
    
    def train(self, save_path: str = None) -> Dict[str, float]:
        """Full training loop with best model saving."""
        logger.info(f"Training on device: {self.device}")
        logger.info(f"Learning rate: {self.learning_rate}")
        logger.info(f"Batch size: {self.batch_size}")
        logger.info(f"Epochs: {self.num_epochs}")
        
        best_f1 = 0
        best_metrics = None
        best_model_state = None
        
        for epoch in range(self.num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            
            # Train
            train_loss = self.train_epoch()
            logger.info(f"Training loss: {train_loss:.4f}")
            
            # Evaluate
            metrics, _, _ = self.evaluate()
            logger.info(f"Validation metrics:")
            for key, value in metrics.items():
                logger.info(f"  {key}: {value:.4f}")
            
            # Track and save best model
            if metrics['f1_weighted'] > best_f1:
                best_f1 = metrics['f1_weighted']
                best_metrics = metrics
                # Save best model state
                best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                logger.info(f"  ✓ New best model! F1: {best_f1:.4f}")
        
        # Load best model state back
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            logger.info(f"\n✓ Loaded best model from training (F1: {best_f1:.4f})")
            
            # Save to file if path provided
            if save_path:
                torch.save(best_model_state, save_path)
                logger.info(f"✓ Best model saved to {save_path}")
        
        return best_metrics


def freeze_layers(model, num_layers_to_freeze: int):
    """Freeze the first N transformer layers."""
    if num_layers_to_freeze > 0:
        # Freeze embeddings
        for param in model.distilbert.embeddings.parameters():
            param.requires_grad = False
        
        # Freeze transformer layers
        for i in range(num_layers_to_freeze):
            if i < len(model.distilbert.transformer.layer):
                for param in model.distilbert.transformer.layer[i].parameters():
                    param.requires_grad = False
        
        logger.info(f"Froze {num_layers_to_freeze} transformer layers")
