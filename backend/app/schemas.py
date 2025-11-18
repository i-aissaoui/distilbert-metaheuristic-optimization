"""Pydantic schemas for request/response validation."""
from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, List, Optional


class PredictionRequest(BaseModel):
    """Request schema for text classification."""
    text: str = Field(..., min_length=1, max_length=5000, description="Text statement to classify")
    use_optimized: bool = Field(default=True, description="Use optimized model if True, baseline if False")
    model_type: Optional[str] = Field(default="pso", description="Which optimized model to use: pso, ga, or bayesian")


class PredictionResponse(BaseModel):
    """Response schema for classification results."""
    model_config = ConfigDict(protected_namespaces=())
    
    text: str
    predicted_label: str
    confidence: float
    all_probabilities: Dict[str, float]
    model_used: str


class ModelInfo(BaseModel):
    """Model information and performance metrics."""
    model_config = ConfigDict(protected_namespaces=())
    
    model_name: str
    num_labels: int
    labels: List[str]
    baseline_metrics: Optional[Dict[str, float]] = None
    optimized_metrics: Optional[Dict[str, float]] = None
    improvement: Optional[Dict[str, float]] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    models_loaded: Dict[str, bool]


class OptimizationConfig(BaseModel):
    """Configuration for selecting which hyperparameters to optimize vs fix."""
    optimize: Optional[Dict[str, bool]] = None
    fixed: Optional[Dict[str, float]] = None
