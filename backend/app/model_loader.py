"""Model loading and inference utilities."""
import os
import warnings

# Suppress ALL warnings before any imports
warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch
import json
from pathlib import Path
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from typing import Dict, Optional, Tuple
import logging

from .config import (
    MODEL_NAME, BASELINE_MODEL_PATH, OPTIMIZED_MODEL_PATH,
    PERFORMANCE_PATH, NUM_LABELS, ID_TO_LABEL, MAX_LENGTH
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelManager:
    """Manages model loading and inference."""
    
    def __init__(self):
        self.tokenizer: Optional[DistilBertTokenizer] = None
        self.baseline_model: Optional[DistilBertForSequenceClassification] = None
        self.optimized_model: Optional[DistilBertForSequenceClassification] = None
        self.performance_metrics: Optional[Dict] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def load_models(self):
        """Load tokenizer and models."""
        logger.info(f"Loading models on device: {self.device}")
        
        # Load tokenizer (use local cache, don't check online)
        try:
            self.tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
            logger.info("Tokenizer loaded")
        except Exception as e:
            logger.warning(f"Failed to load tokenizer: {e}")
            logger.info("Running in DEMO mode without actual models")
        
        # Load baseline model if exists
        if BASELINE_MODEL_PATH.exists():
            try:
                self.baseline_model = DistilBertForSequenceClassification.from_pretrained(
                    MODEL_NAME,
                    num_labels=NUM_LABELS,
                    local_files_only=True,
                    ignore_mismatched_sizes=True
                )
                state_dict = torch.load(BASELINE_MODEL_PATH, map_location=self.device)
                self.baseline_model.load_state_dict(state_dict)
                self.baseline_model.to(self.device)
                self.baseline_model.eval()
                logger.info("Baseline model loaded")
            except Exception as e:
                logger.warning(f"Failed to load baseline model: {e}")
                logger.info("Baseline model not available - using demo mode")
        else:
            logger.info("Baseline model not found - using demo mode")
        
        # Load optimized model if exists (optional - from previous optimization runs)
        if OPTIMIZED_MODEL_PATH.exists():
            try:
                self.optimized_model = DistilBertForSequenceClassification.from_pretrained(
                    MODEL_NAME,
                    num_labels=NUM_LABELS,
                    local_files_only=True,
                    ignore_mismatched_sizes=True
                )
                state_dict = torch.load(OPTIMIZED_MODEL_PATH, map_location=self.device)
                self.optimized_model.load_state_dict(state_dict)
                self.optimized_model.to(self.device)
                self.optimized_model.eval()
                logger.info("Previously optimized model loaded")
            except Exception as e:
                pass  # Silently skip if not available
        
        # Load performance metrics
        if PERFORMANCE_PATH.exists():
            try:
                with open(PERFORMANCE_PATH, 'r') as f:
                    self.performance_metrics = json.load(f)
                logger.info("Performance metrics loaded")
            except Exception as e:
                logger.warning(f"Failed to load performance metrics: {e}")
    
    def predict(self, text: str, use_optimized: bool = True) -> Tuple[str, float, Dict[str, float], str]:
        """
        Make prediction on input text.
        
        Returns:
            Tuple of (predicted_label, confidence, all_probabilities, model_used)
        """
        # Select model
        if use_optimized and self.optimized_model is not None:
            model = self.optimized_model
            model_name = "optimized"
        elif self.baseline_model is not None:
            model = self.baseline_model
            model_name = "baseline"
        else:
            # DEMO MODE: Generate random predictions for demonstration
            logger.info("Running in DEMO mode - generating sample predictions")
            return self._demo_predict(text, use_optimized)
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)[0]
            predicted_class = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_class].item()
        
        # Format results
        predicted_label = ID_TO_LABEL[predicted_class]
        all_probs = {
            ID_TO_LABEL[i]: float(probabilities[i])
            for i in range(NUM_LABELS)
        }
        
        return predicted_label, confidence, all_probs, model_name
    
    def _demo_predict(self, text: str, use_optimized: bool) -> Tuple[str, float, Dict[str, float], str]:
        """Generate demo predictions when models are not available."""
        import random
        import hashlib
        
        # Use text hash for consistent predictions
        text_hash = int(hashlib.md5(text.encode()).hexdigest(), 16)
        random.seed(text_hash)
        
        # Generate probabilities
        probs = [random.random() for _ in range(NUM_LABELS)]
        total = sum(probs)
        probs = [p / total for p in probs]
        
        # Adjust for optimized model (slightly higher confidence)
        if use_optimized:
            max_idx = probs.index(max(probs))
            probs[max_idx] *= 1.15
            total = sum(probs)
            probs = [p / total for p in probs]
        
        predicted_class = probs.index(max(probs))
        predicted_label = ID_TO_LABEL[predicted_class]
        confidence = probs[predicted_class]
        
        all_probs = {
            ID_TO_LABEL[i]: probs[i]
            for i in range(NUM_LABELS)
        }
        
        model_name = "optimized (demo)" if use_optimized else "baseline (demo)"
        
        return predicted_label, confidence, all_probs, model_name
    
    def get_model_info(self) -> Dict:
        """Get model information and performance metrics."""
        info = {
            "model_name": MODEL_NAME,
            "num_labels": NUM_LABELS,
            "labels": list(ID_TO_LABEL.values()),
            "baseline_loaded": self.baseline_model is not None,
            "optimized_loaded": self.optimized_model is not None,
        }
        
        if self.performance_metrics:
            info.update(self.performance_metrics)
        
        return info
    
    def is_ready(self) -> Dict[str, bool]:
        """Check if models are loaded."""
        return {
            "baseline": self.baseline_model is not None,
            "optimized": self.optimized_model is not None,
        }


# Global model manager instance
model_manager = ModelManager()
