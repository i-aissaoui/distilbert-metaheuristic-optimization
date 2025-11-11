"""Configuration settings for the application."""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

# Model settings
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128
NUM_LABELS = 6

# Label mapping for LIAR dataset
LABEL_MAP = {
    0: "pants-fire",
    1: "false",
    2: "barely-true",
    3: "half-true",
    4: "mostly-true",
    5: "true"
}

ID_TO_LABEL = LABEL_MAP
LABEL_TO_ID = {v: k for k, v in LABEL_MAP.items()}

# Model paths
BASELINE_MODEL_PATH = MODELS_DIR / "baseline_model.pt"
OPTIMIZED_MODEL_PATH = MODELS_DIR / "optimized_model.pt"
PERFORMANCE_PATH = MODELS_DIR / "performance_comparison.json"

# API settings
API_TITLE = "Fake News Detection API"
API_VERSION = "1.0.0"
API_DESCRIPTION = "AI-powered fake news detection using PSO-optimized DistilBERT"

# Device configuration
DEVICE = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
