"""FastAPI application for fake news detection."""
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import logging
import json
from pathlib import Path
import asyncio
from typing import List
import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')

from .schemas import (
    PredictionRequest, PredictionResponse,
    ModelInfo, HealthResponse
)
from .model_loader import model_manager
from .config import API_TITLE, API_VERSION, API_DESCRIPTION, ID_TO_LABEL, BASE_DIR

# Configure logging - only show important messages
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manage WebSocket connections for real-time updates."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting message: {e}")


manager = ConnectionManager()

# Initialize FastAPI app
app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESCRIPTION
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global training status tracker
training_status = {
    "status": "idle",
    "progress": 0,
    "current_step": 0,
    "total_steps": 0,
    "current_epoch": 0,
    "total_epochs": 0,
    "loss": 0.0,
    "message": ""
}


@app.on_event("startup")
async def startup_event():
    """Load models on startup."""
    logger.info("Starting up application...")
    try:
        model_manager.load_models()
        logger.info("Models loaded successfully")
    except Exception as e:
        logger.error(f"Error loading models: {e}")


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "message": "Fake News Detection API",
        "version": API_VERSION,
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    models_status = model_manager.is_ready()
    return HealthResponse(
        status="healthy" if any(models_status.values()) else "no_models_loaded",
        models_loaded=models_status
    )


@app.get("/labels", tags=["Info"])
async def get_labels():
    """Get available classification labels."""
    return {
        "labels": list(ID_TO_LABEL.values()),
        "descriptions": {
            "pants-fire": "Completely false statement",
            "false": "False statement",
            "barely-true": "Statement with minimal truth",
            "half-true": "Partially true statement",
            "mostly-true": "Largely true statement",
            "true": "Completely true statement"
        }
    }


@app.get("/model-info", response_model=ModelInfo, tags=["Info"])
async def get_model_info():
    """Get model information and performance metrics."""
    try:
        info = model_manager.get_model_info()
        return ModelInfo(**info)
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest):
    """
    Classify a text statement for truthfulness.
    
    - **text**: The statement to classify
    - **use_optimized**: Whether to use the PSO-optimized model (default: True)
    """
    try:
        # Make prediction
        predicted_label, confidence, all_probs, model_used = model_manager.predict(
            request.text,
            use_optimized=request.use_optimized
        )
        
        return PredictionResponse(
            text=request.text,
            predicted_label=predicted_label,
            confidence=confidence,
            all_probabilities=all_probs,
            model_used=model_used
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during prediction")


@app.get("/training-history", tags=["Training"])
async def get_training_history():
    """Get training history for all models."""
    try:
        log_file = BASE_DIR / "logs" / "training_history.json"
        if log_file.exists():
            with open(log_file, 'r') as f:
                return json.load(f)
        return {"message": "No training history available", "data": []}
    except Exception as e:
        logger.error(f"Error loading training history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/pso-history", tags=["Training"])
async def get_pso_history():
    """Get PSO optimization history."""
    try:
        log_file = BASE_DIR / "logs" / "pso_history.json"
        if log_file.exists():
            with open(log_file, 'r') as f:
                return json.load(f)
        return {"message": "No PSO history available", "data": []}
    except Exception as e:
        logger.error(f"Error loading PSO history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/pso-animation", tags=["Training"])
async def get_pso_animation():
    """Get PSO particle animation data."""
    try:
        # Try new organized structure first
        log_file = BASE_DIR / "logs" / "results" / "pso" / "animation.json"
        if not log_file.exists():
            # Fallback to old location
            log_file = BASE_DIR / "logs" / "pso_animation.json"
        
        if log_file.exists():
            with open(log_file, 'r') as f:
                data = json.load(f)
                
                # Normalize data format - ensure all particles have parameters at top level
                if 'data' in data:
                    for particle in data['data']:
                        # If params are nested, move them to top level
                        if 'params' in particle and not 'learning_rate' in particle:
                            particle['learning_rate'] = particle['params'].get('learning_rate')
                            particle['batch_size'] = particle['params'].get('batch_size')
                            particle['dropout'] = particle['params'].get('dropout')
                            particle['frozen_layers'] = particle['params'].get('frozen_layers')
                
                return data
        return {"message": "No PSO animation data available", "data": []}
    except Exception as e:
        logger.error(f"Error loading PSO animation data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ga-history", tags=["Training"])
async def get_ga_history():
    """Get GA optimization history."""
    try:
        log_file = BASE_DIR / "logs" / "ga_history.json"
        if log_file.exists():
            with open(log_file, 'r') as f:
                return json.load(f)
        return {"message": "No GA history available", "data": []}
    except Exception as e:
        logger.error(f"Error loading GA history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ga-animation", tags=["Training"])
async def get_ga_animation():
    """Get GA individual animation data."""
    try:
        log_file = BASE_DIR / "logs" / "ga_animation.json"
        if log_file.exists():
            with open(log_file, 'r') as f:
                return json.load(f)
        return {"message": "No GA animation data available", "data": []}
    except Exception as e:
        logger.error(f"Error loading GA animation data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/bayesian-history", tags=["Training"])
async def get_bayesian_history():
    """Get Bayesian optimization history."""
    try:
        log_file = BASE_DIR / "logs" / "bayesian_history.json"
        if log_file.exists():
            with open(log_file, 'r') as f:
                return json.load(f)
        return {"message": "No Bayesian history available", "data": []}
    except Exception as e:
        logger.error(f"Error loading Bayesian history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/bayesian-animation", tags=["Training"])
async def get_bayesian_animation():
    """Get Bayesian trial animation data."""
    try:
        log_file = BASE_DIR / "logs" / "bayesian_animation.json"
        if log_file.exists():
            with open(log_file, 'r') as f:
                return json.load(f)
        return {"message": "No Bayesian animation data available", "data": []}
    except Exception as e:
        logger.error(f"Error loading Bayesian animation data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/hyperparameters", tags=["Training"])
async def get_hyperparameters():
    """Get hyperparameters used for baseline and optimized models."""
    try:
        baseline_params = {
            "learning_rate": 2e-5,
            "batch_size": 16,
            "dropout": 0.1,
            "frozen_layers": 0,
            "epochs": 3
        }
        
        optimized_params = {}
        params_file = BASE_DIR / "models" / "best_hyperparameters.json"
        if params_file.exists():
            with open(params_file, 'r') as f:
                optimized_params = json.load(f)
        
        return {
            "baseline": baseline_params,
            "optimized": optimized_params if optimized_params else None
        }
    except Exception as e:
        logger.error(f"Error loading hyperparameters: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/start-training", tags=["Training"])
async def start_training(
    learning_rate: float = 2e-5,
    batch_size: int = 16,
    dropout: float = 0.1,
    frozen_layers: int = 0,
    epochs: int = 3
):
    """Start training with custom parameters."""
    try:
        import subprocess
        import sys
        
        logger.info(f"🚀 Starting custom training with parameters:")
        logger.info(f"   Learning Rate: {learning_rate}")
        logger.info(f"   Batch Size: {batch_size}")
        logger.info(f"   Dropout: {dropout}")
        logger.info(f"   Frozen Layers: {frozen_layers}")
        logger.info(f"   Epochs: {epochs}")
        
        # Start training in background with custom parameters
        cmd = [
            sys.executable,
            "training/train_baseline.py",
            "--learning_rate", str(learning_rate),
            "--batch_size", str(batch_size),
            "--dropout", str(dropout),
            "--frozen_layers", str(frozen_layers),
            "--epochs", str(epochs)
        ]
        
        process = subprocess.Popen(
            cmd,
            cwd=str(BASE_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        logger.info(f"✅ Training process started with PID: {process.pid}")
        
        # Reset training status at start
        global training_status
        import time
        start_time = time.time()
        training_status = {
            "status": "running",
            "progress": 0,
            "current_step": 0,
            "total_steps": 0,
            "current_epoch": 0,
            "total_epochs": 0,
            "loss": 0.0,
            "message": "Starting training...",
            "start_time": start_time
        }
        
        # Log output in real-time and parse progress (non-blocking)
        import threading
        import re
        def log_output():
            global training_status
            for line in process.stdout:
                logger.info(f"[Training] {line.strip()}")
                
                # Parse epoch info: "Epoch 2/3"
                epoch_match = re.search(r'Epoch\s+(\d+)/(\d+)', line)
                if epoch_match:
                    current_epoch = int(epoch_match.group(1))
                    total_epochs = int(epoch_match.group(2))
                    training_status["current_epoch"] = current_epoch
                    training_status["total_epochs"] = total_epochs
                
                # Parse progress from tqdm output: "Training:  42%|████▏     | 268/642 [00:33<00:45,  8.13it/s, loss=1.77]"
                match = re.search(r'Training:\s+(\d+)%.*?\|\s+(\d+)/(\d+).*?loss=([\d.]+)', line)
                if match:
                    progress_pct = int(match.group(1))  # Progress within current epoch
                    current = int(match.group(2))
                    total = int(match.group(3))
                    loss = float(match.group(4))
                    
                    # Calculate overall progress across all epochs
                    current_epoch = training_status.get('current_epoch', 1)
                    total_epochs = training_status.get('total_epochs', 3)
                    
                    # Overall progress = (completed epochs + current epoch progress) / total epochs
                    epoch_progress = (current_epoch - 1) / total_epochs  # Completed epochs
                    current_epoch_progress = (progress_pct / 100) / total_epochs  # Current epoch contribution
                    overall_progress = int((epoch_progress + current_epoch_progress) * 100)
                    
                    training_status.update({
                        "status": "running",
                        "progress": overall_progress,
                        "current_step": current,
                        "total_steps": total,
                        "loss": loss,
                        "message": f"Epoch {current_epoch}/{total_epochs} - Step {current}/{total}"
                    })
                
                # Parse final metrics: "INFO:__main__:Final Accuracy: 0.9928"
                if "Final Accuracy:" in line:
                    acc_match = re.search(r'Final Accuracy:\s+([\d.]+)', line)
                    if acc_match:
                        accuracy = float(acc_match.group(1))
                        training_status["accuracy"] = accuracy
                        logger.info(f"✅ Final Accuracy captured: {accuracy}")
                
                if "Final F1 Score:" in line:
                    f1_match = re.search(r'Final F1 Score:\s+([\d.]+)', line)
                    if f1_match:
                        f1_score = float(f1_match.group(1))
                        training_status["f1_score"] = f1_score
                        logger.info(f"✅ Final F1 Score captured: {f1_score}")
                        
                        # When we get final F1, training is done
                        end_time = time.time()
                        training_time = end_time - training_status.get("start_time", end_time)
                        
                        # Load all metrics from baseline_metrics.json
                        try:
                            metrics_file = BASE_DIR / "models" / "baseline_metrics.json"
                            if metrics_file.exists():
                                with open(metrics_file, 'r') as f:
                                    metrics = json.load(f)
                                    training_status["accuracy"] = metrics.get("accuracy", training_status.get("accuracy"))
                                    training_status["f1_score"] = metrics.get("f1_weighted", f1_score)
                                    training_status["f1_weighted"] = metrics.get("f1_weighted")
                                    training_status["f1_macro"] = metrics.get("f1_macro")
                                    training_status["precision"] = metrics.get("precision")
                                    training_status["recall"] = metrics.get("recall")
                                    training_status["loss"] = metrics.get("loss")
                        except Exception as e:
                            logger.error(f"Error loading metrics: {e}")
                        
                        training_status["status"] = "completed"
                        training_status["progress"] = 100
                        training_status["message"] = f"Training completed - Accuracy: {training_status.get('accuracy', 0):.4f}, F1: {training_status.get('f1_score', 0):.4f}"
                        training_status["training_time"] = training_time
                        logger.info(f"✅ Training marked as completed with all metrics")
                
                # Check if training completed (additional patterns)
                if "Training Complete" in line or "Baseline Training Complete" in line or "Optimization Complete" in line:
                    end_time = time.time()
                    training_time = end_time - training_status.get("start_time", end_time)
                    training_status["status"] = "completed"
                    training_status["progress"] = 100
                    training_status["message"] = "Training completed successfully"
                    training_status["training_time"] = training_time
        
        thread = threading.Thread(target=log_output, daemon=True)
        thread.start()
        
        return {
            "status": "started",
            "message": "Training started in background",
            "pid": process.pid,
            "parameters": {
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "dropout": dropout,
                "frozen_layers": frozen_layers,
                "epochs": epochs
            }
        }
    except Exception as e:
        logger.error(f"❌ Error starting training: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/start-bayesian", tags=["Training"])
async def start_bayesian(
    n_trials: int = 10
):
    """Start Bayesian Optimization (Optuna)."""
    try:
        import subprocess
        import sys
        
        logger.info(f"📊 Starting Bayesian Optimization with parameters:")
        logger.info(f"   Number of Trials: {n_trials}")
        
        # Clear full data results when starting new optimization
        full_data_file = BASE_DIR / "logs" / "results" / "bayesian" / "full_data.json"
        if full_data_file.exists():
            full_data_file.unlink()
            logger.info("🗑️  Cleared previous full data results")
        
        # Start Bayesian optimization in background with parameters
        cmd = [
            sys.executable,
            "training/bayesian_optimization.py",
            str(n_trials)
        ]
        
        process = subprocess.Popen(
            cmd,
            cwd=str(BASE_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        logger.info(f"✅ Bayesian optimization process started with PID: {process.pid}")
        
        # Log output in real-time (non-blocking) - filter to reduce noise
        import threading
        def log_output():
            for line in process.stdout:
                # Only log important Bayesian messages
                if any(keyword in line for keyword in ['Trial', 'Best', 'Complete', 'Starting', 'ERROR', 'WARNING']):
                    logger.info(f"[Bayesian] {line.strip()}")
        
        thread = threading.Thread(target=log_output, daemon=True)
        thread.start()
        
        return {
            "status": "started",
            "message": "Bayesian optimization started in background",
            "pid": process.pid,
            "parameters": {
                "n_trials": n_trials
            }
        }
    except Exception as e:
        logger.error(f"❌ Error starting Bayesian optimization: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/start-ga", tags=["Training"])
async def start_ga(
    population_size: int = 5,
    num_generations: int = 10
):
    """Start Genetic Algorithm optimization."""
    try:
        import subprocess
        import sys
        
        logger.info(f"🧬 Starting Genetic Algorithm optimization with parameters:")
        logger.info(f"   Population Size: {population_size}")
        logger.info(f"   Num Generations: {num_generations}")
        
        # Clear full data results when starting new optimization
        full_data_file = BASE_DIR / "logs" / "results" / "ga" / "full_data.json"
        if full_data_file.exists():
            full_data_file.unlink()
            logger.info("🗑️  Cleared previous full data results")
        
        # Start GA in background with parameters
        cmd = [
            sys.executable,
            "training/ga_optimization.py",
            str(population_size),
            str(num_generations)
        ]
        
        process = subprocess.Popen(
            cmd,
            cwd=str(BASE_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        logger.info(f"✅ GA process started with PID: {process.pid}")
        
        # Log output in real-time (non-blocking) - filter to reduce noise
        import threading
        def log_output():
            for line in process.stdout:
                # Only log important GA messages
                if any(keyword in line for keyword in ['Generation', 'Best', 'Complete', 'Starting', 'ERROR', 'WARNING']):
                    logger.info(f"[GA] {line.strip()}")
        
        thread = threading.Thread(target=log_output, daemon=True)
        thread.start()
        
        return {
            "status": "started",
            "message": "Genetic Algorithm optimization started in background",
            "pid": process.pid,
            "parameters": {
                "population_size": population_size,
                "num_generations": num_generations
            }
        }
    except Exception as e:
        logger.error(f"❌ Error starting GA: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/start-pso", tags=["Training"])
async def start_pso(
    swarmsize: int = 5,
    maxiter: int = 10
):
    """Start PSO optimization."""
    try:
        import subprocess
        import sys
        
        logger.info(f"🐝 Starting PSO optimization with parameters:")
        logger.info(f"   Swarm Size: {swarmsize}")
        logger.info(f"   Max Iterations: {maxiter}")
        
        # Clear full data results when starting new optimization
        full_data_file = BASE_DIR / "logs" / "results" / "pso" / "full_data.json"
        if full_data_file.exists():
            full_data_file.unlink()
            logger.info("🗑️  Cleared previous full data results")
        
        # Start PSO in background with parameters
        cmd = [
            sys.executable,
            "training/pso_optimization.py",
            str(swarmsize),
            str(maxiter)
        ]
        
        process = subprocess.Popen(
            cmd,
            cwd=str(BASE_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        logger.info(f"✅ PSO process started with PID: {process.pid}")
        
        # Log output in real-time (non-blocking) - filter to reduce noise
        import threading
        def log_output():
            for line in process.stdout:
                # Only log important PSO messages, skip individual particle training details
                if any(keyword in line for keyword in ['Iteration', 'Best', 'Complete', 'Starting', 'ERROR', 'WARNING']):
                    logger.info(f"[PSO] {line.strip()}")
        
        thread = threading.Thread(target=log_output, daemon=True)
        thread.start()
        
        return {
            "status": "started",
            "message": "PSO optimization started in background",
            "pid": process.pid,
            "parameters": {
                "swarmsize": swarmsize,
                "maxiter": maxiter
            }
        }
    except Exception as e:
        logger.error(f"❌ Error starting PSO: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train-full-data", tags=["Training"])
async def train_on_full_data(
    algorithm: str,
    learning_rate: float,
    batch_size: int,
    dropout: float,
    frozen_layers: int,
    epochs: int = 3
):
    """Train a final model on full data using optimized parameters from an algorithm."""
    try:
        import subprocess
        import sys
        
        logger.info(f"🚀 Starting FULL DATA training with {algorithm.upper()} optimized parameters:")
        logger.info(f"   Learning Rate: {learning_rate}")
        logger.info(f"   Batch Size: {batch_size}")
        logger.info(f"   Dropout: {dropout}")
        logger.info(f"   Frozen Layers: {frozen_layers}")
        logger.info(f"   Epochs: {epochs}")
        
        # Start training on FULL data with optimized parameters
        cmd = [
            sys.executable,
            "training/train_baseline.py",
            "--learning_rate", str(learning_rate),
            "--batch_size", str(batch_size),
            "--dropout", str(dropout),
            "--frozen_layers", str(frozen_layers),
            "--epochs", str(epochs)
        ]
        
        process = subprocess.Popen(
            cmd,
            cwd=str(BASE_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        logger.info(f"✅ Full data training process started with PID: {process.pid}")
        
        # Reset training status at start
        global training_status
        import time
        start_time = time.time()
        training_status = {
            "status": "running",
            "progress": 0,
            "current_step": 0,
            "total_steps": 0,
            "current_epoch": 0,
            "total_epochs": epochs,
            "loss": 0.0,
            "message": f"Training on full data with {algorithm.upper()} parameters...",
            "start_time": start_time
        }
        
        # Log output in background and parse progress
        import threading
        import re
        def log_output():
            global training_status
            for line in process.stdout:
                logger.info(f"[{algorithm.upper()} Full Training] {line.strip()}")
                
                # Parse epoch info
                epoch_match = re.search(r'Epoch\s+(\d+)/(\d+)', line)
                if epoch_match:
                    current_epoch = int(epoch_match.group(1))
                    total_epochs = int(epoch_match.group(2))
                    training_status["current_epoch"] = current_epoch
                    training_status["total_epochs"] = total_epochs
                
                # Parse progress from tqdm
                match = re.search(r'Training:\s+(\d+)%.*?\|\s+(\d+)/(\d+).*?loss=([\d.]+)', line)
                if match:
                    progress_pct = int(match.group(1))
                    current = int(match.group(2))
                    total = int(match.group(3))
                    loss = float(match.group(4))
                    
                    # Calculate overall progress
                    current_epoch = training_status.get('current_epoch', 1)
                    total_epochs = training_status.get('total_epochs', epochs)
                    
                    epoch_progress = (current_epoch - 1) / total_epochs
                    current_epoch_progress = (progress_pct / 100) / total_epochs
                    overall_progress = int((epoch_progress + current_epoch_progress) * 100)
                    
                    training_status.update({
                        "status": "running",
                        "progress": overall_progress,
                        "current_step": current,
                        "total_steps": total,
                        "loss": loss,
                        "message": f"Training on full data - Epoch {current_epoch}/{total_epochs}"
                    })
                
                # Check if training completed
                if "Training Complete" in line or "Final Accuracy" in line:
                    training_status.update({
                        "status": "completed",
                        "progress": 100,
                        "message": "Training completed successfully!"
                    })
                
                # Parse final metrics
                acc_match = re.search(r'Final Accuracy:\s+([\d.]+)', line)
                f1_match = re.search(r'Final F1 Score:\s+([\d.]+)', line)
                if acc_match:
                    final_accuracy = float(acc_match.group(1))
                    training_status["final_accuracy"] = final_accuracy
                if f1_match:
                    final_f1 = float(f1_match.group(1))
                    training_status["final_f1"] = final_f1
                    
                    # Save full data training results to algorithm's folder
                    result_dir = BASE_DIR / "logs" / "results" / algorithm
                    result_dir.mkdir(parents=True, exist_ok=True)
                    
                    full_data_result = {
                        "type": f"{algorithm}_full_data",
                        "algorithm": algorithm,
                        "completedAt": __import__('datetime').datetime.now().isoformat(),
                        "accuracy": final_accuracy if 'final_accuracy' in training_status else final_f1,
                        "f1_score": final_f1,
                        "message": f"Full data training with {algorithm.upper()} parameters complete",
                        "parameters": {
                            "learning_rate": learning_rate,
                            "batch_size": batch_size,
                            "dropout": dropout,
                            "frozen_layers": frozen_layers,
                            "epochs": epochs
                        },
                        "training_type": "full_data"
                    }
                    
                    # Save to full_data.json
                    with open(result_dir / "full_data.json", 'w') as f:
                        json.dump(full_data_result, f, indent=2)
                    
                    logger.info(f"✅ Saved full data training results to {result_dir / 'full_data.json'}")
        
        thread = threading.Thread(target=log_output, daemon=True)
        thread.start()
        
        return {
            "status": "started",
            "message": f"Training on full data with {algorithm.upper()} optimized parameters",
            "pid": process.pid,
            "algorithm": algorithm
        }
    except Exception as e:
        logger.error(f"❌ Error starting full data training: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/training")
async def websocket_training(websocket: WebSocket):
    """WebSocket endpoint for real-time training updates."""
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive and listen for messages
            data = await websocket.receive_text()
            # Echo back or handle commands if needed
            await websocket.send_json({"status": "connected"})
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("WebSocket disconnected")


@app.get("/training-status", tags=["Training"])
async def get_training_status():
    """Get current training status."""
    try:
        global training_status
        
        # If global status is running, return it (custom training)
        if training_status.get("status") == "running":
            return training_status
        
        # Check organized status files for PSO/GA/Bayesian
        # Find the most recently modified status file that's running
        running_statuses = []
        for algorithm in ['pso', 'ga', 'bayesian']:
            status_file = BASE_DIR / "logs" / "results" / algorithm / "status.json"
            if status_file.exists():
                with open(status_file, 'r') as f:
                    file_status = json.load(f)
                    # Only consider running statuses (not completed ones with 100% progress)
                    if file_status.get("status") == "running" and file_status.get("progress", 0) < 100:
                        running_statuses.append({
                            'status': file_status,
                            'modified_time': status_file.stat().st_mtime
                        })
        
        # Return the most recently modified running status
        if running_statuses:
            most_recent = max(running_statuses, key=lambda x: x['modified_time'])
            return most_recent['status']
        
        # If no running status, check for any status file (including completed)
        for algorithm in ['pso', 'ga', 'bayesian']:
            status_file = BASE_DIR / "logs" / "results" / algorithm / "status.json"
            if status_file.exists():
                with open(status_file, 'r') as f:
                    return json.load(f)
        
        return training_status
    except Exception as e:
        logger.error(f"Error loading training status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/save-result", tags=["Training"])
async def save_training_result(data: dict):
    """Save training result to JSON file (latest.json only)."""
    try:
        result_type = data.get("type", "custom")
        
        # Create organized folder structure: logs/results/{algorithm}/
        result_dir = BASE_DIR / "logs" / "results" / result_type
        result_dir.mkdir(parents=True, exist_ok=True)
        
        # Save only as latest.json (overwrites previous)
        latest_file = result_dir / "latest.json"
        with open(latest_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"✅ Saved {result_type} training result to latest.json")
        
        return {
            "status": "success", 
            "message": f"Result saved for {result_type}",
            "file": str(latest_file.relative_to(BASE_DIR))
        }
    except Exception as e:
        logger.error(f"Error saving result: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/load-result/{algorithm}/{file_name}", tags=["Training"])
async def load_training_result(algorithm: str, file_name: str = "latest"):
    """Load training result for an algorithm. Returns null if not found."""
    try:
        # Construct file path
        result_file = BASE_DIR / "logs" / "results" / algorithm / f"{file_name}.json"
        if result_file.exists() and result_file.stat().st_size > 0:
            with open(result_file, 'r') as f:
                content = f.read().strip()
                if content:
                    return json.loads(content)
        
        # Return null if file doesn't exist or is empty
        return None
    except json.JSONDecodeError:
        # File exists but invalid JSON - return None silently
        return None
    except Exception as e:
        logger.debug(f"Could not load result for {algorithm}/{file_name}: {e}")
        return None

@app.get("/load-result/{algorithm}", tags=["Training"])
async def load_training_result_default(algorithm: str):
    """Load latest training result for an algorithm. Returns null if not found."""
    return await load_training_result(algorithm, "latest")


@app.post("/clear-history", tags=["Training"])
async def clear_training_history():
    """Clear all training history and results."""
    try:
        results_dir = BASE_DIR / "logs" / "results"
        if results_dir.exists():
            for file in results_dir.glob("*.json"):
                file.unlink()
        
        return {"status": "success", "message": "Training history cleared"}
    except Exception as e:
        logger.error(f"Error clearing history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
