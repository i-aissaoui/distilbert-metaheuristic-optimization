"""Training logger to save metrics and PSO particle positions for visualization."""
import json
import os
from pathlib import Path
from typing import Dict, List, Any
import numpy as np


class TrainingLogger:
    """Logger for training metrics and PSO visualization data."""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        self.training_history = []
        self.pso_history = []
        self.particle_positions = []
        
    def log_epoch(self, epoch: int, metrics: Dict[str, float], model_type: str = "baseline"):
        """Log metrics for a single epoch."""
        entry = {
            "epoch": epoch,
            "model_type": model_type,
            "timestamp": str(np.datetime64('now')),
            **metrics
        }
        self.training_history.append(entry)
        
    def log_pso_iteration(self, iteration: int, particles: List[Dict[str, Any]], 
                         best_score: float, best_params: Dict[str, Any]):
        """Log PSO iteration data including particle positions."""
        entry = {
            "iteration": iteration,
            "best_score": best_score,
            "best_params": best_params,
            "particles": particles,
            "timestamp": str(np.datetime64('now'))
        }
        self.pso_history.append(entry)
        
        # Store particle positions for animation
        positions = []
        for p in particles:
            positions.append({
                "learning_rate": p.get("learning_rate", 0),
                "batch_size": p.get("batch_size", 0),
                "dropout": p.get("dropout", 0),
                "frozen_layers": p.get("frozen_layers", 0),
                "score": p.get("score", 0)
            })
        self.particle_positions.append({
            "iteration": iteration,
            "positions": positions
        })
    
    def save_training_history(self, filename: str = "training_history.json"):
        """Save training history to JSON file."""
        filepath = self.log_dir / filename
        with open(filepath, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        print(f"Training history saved to {filepath}")
    
    def save_pso_history(self, filename: str = "pso_history.json"):
        """Save PSO history to JSON file."""
        filepath = self.log_dir / filename
        with open(filepath, 'w') as f:
            json.dump(self.pso_history, f, indent=2)
        print(f"PSO history saved to {filepath}")
    
    def save_particle_animation_data(self, filename: str = "pso_animation.json"):
        """Save particle positions for animation."""
        filepath = self.log_dir / filename
        with open(filepath, 'w') as f:
            json.dump(self.particle_positions, f, indent=2)
        print(f"PSO animation data saved to {filepath}")
    
    def save_all(self):
        """Save all logged data."""
        self.save_training_history()
        self.save_pso_history()
        self.save_particle_animation_data()


# Global logger instance
logger = TrainingLogger(log_dir="backend/logs")
