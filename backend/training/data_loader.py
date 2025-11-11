"""Data loading utilities for spam dataset."""
import pandas as pd
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import DistilBertTokenizer
from typing import Dict, Tuple
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SpamDataset(Dataset):
    """PyTorch Dataset for spam classification."""
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': label
        }


def load_spam_dataset() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the hate speech dataset from text files.
    
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    logger.info("Loading hate speech dataset...")
    
    try:
        # Load from separate text files
        data_dir = Path(__file__).parent.parent / "data"
        
        # Load training data
        with open(data_dir / "train_text.txt", 'r', encoding='utf-8') as f:
            train_texts = [line.strip() for line in f.readlines()]
        with open(data_dir / "train_labels.txt", 'r', encoding='utf-8') as f:
            train_labels = [int(line.strip()) for line in f.readlines()]
        
        # Load validation data
        with open(data_dir / "val_text.txt", 'r', encoding='utf-8') as f:
            val_texts = [line.strip() for line in f.readlines()]
        with open(data_dir / "val_labels.txt", 'r', encoding='utf-8') as f:
            val_labels = [int(line.strip()) for line in f.readlines()]
        
        # Load test data
        with open(data_dir / "test_text.txt", 'r', encoding='utf-8') as f:
            test_texts = [line.strip() for line in f.readlines()]
        with open(data_dir / "test_labels.txt", 'r', encoding='utf-8') as f:
            test_labels = [int(line.strip()) for line in f.readlines()]
        
        # Create DataFrames
        train_df = pd.DataFrame({'text': train_texts, 'label': train_labels})
        val_df = pd.DataFrame({'text': val_texts, 'label': val_labels})
        test_df = pd.DataFrame({'text': test_texts, 'label': test_labels})
        
        logger.info(f"Train size: {len(train_df)}")
        logger.info(f"Validation size: {len(val_df)}")
        logger.info(f"Test size: {len(test_df)}")
        logger.info(f"Number of classes: {train_df['label'].nunique()}")
        logger.info(f"Label distribution - Train: {train_df['label'].value_counts().to_dict()}")
        
        return train_df, val_df, test_df
    
    except Exception as e:
        logger.error(f"Error loading hate speech dataset: {e}")
        logger.info("Creating sample dataset for demonstration...")
        
        # Create sample data if dataset loading fails
        sample_data = {
            'text': [
                "I hate you so much",
                "Have a nice day!",
                "You are stupid and ugly",
                "Thanks for your help",
                "Go kill yourself",
            ],
            'label': [1, 0, 1, 0, 1]  # 1=hate, 0=not-hate
        }
        
        train_df = pd.DataFrame(sample_data)
        val_df = train_df.copy()
        test_df = train_df.copy()
        
        logger.warning("Using sample data. Ensure text files are in backend/data/ directory.")
        
        return train_df, val_df, test_df


def prepare_datasets(
    tokenizer: DistilBertTokenizer,
    max_length: int = 128,
    subset_fraction: float = 1.0
) -> Tuple[SpamDataset, SpamDataset, SpamDataset]:
    """
    Prepare train, validation, and test datasets.
    
    Args:
        tokenizer: DistilBERT tokenizer
        max_length: Maximum sequence length
        subset_fraction: Fraction of data to use (0.0-1.0). Use smaller values for faster training.
    
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    train_df, val_df, test_df = load_spam_dataset()
    
    # Use subset of data if specified (for faster training during hyperparameter optimization)
    if subset_fraction < 1.0:
        # IMPORTANT: Use fixed random_state=42 so all algorithms get the SAME subset
        # This ensures fair comparison across PSO, GA, and Bayesian optimization
        train_df = train_df.groupby('label', group_keys=False).apply(
            lambda x: x.sample(frac=subset_fraction, random_state=42)
        ).reset_index(drop=True)
        val_df = val_df.groupby('label', group_keys=False).apply(
            lambda x: x.sample(frac=subset_fraction, random_state=42)
        ).reset_index(drop=True)
        
        logger.info(f"✅ Using {subset_fraction*100:.1f}% of data (random_state=42 for reproducibility)")
        logger.info(f"   Train: {len(train_df)} samples, Val: {len(val_df)} samples")
    
    # Create datasets
    train_dataset = SpamDataset(
        train_df['text'].values,
        train_df['label'].values,
        tokenizer,
        max_length
    )
    
    val_dataset = SpamDataset(
        val_df['text'].values,
        val_df['label'].values,
        tokenizer,
        max_length
    )
    
    test_dataset = SpamDataset(
        test_df['text'].values,
        test_df['label'].values,
        tokenizer,
        max_length
    )
    
    return train_dataset, val_dataset, test_dataset
