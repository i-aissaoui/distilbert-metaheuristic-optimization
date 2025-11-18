# DistilBERT Multi-Algorithm Hyperparameter Optimization

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Next.js-14.0+-black.svg)](https://nextjs.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> A production-ready full-stack application showcasing **three metaheuristic optimization algorithms (PSO, GA, Bayesian)** for hyperparameter tuning of DistilBERT models with real-time visualization.

## 🎯 Overview

This system demonstrates advanced optimization techniques for deep learning hyperparameter tuning. It compares **Particle Swarm Optimization (PSO)**, **Genetic Algorithms (GA)**, and **Bayesian Optimization** on a DistilBERT text classification task. Features include FastAPI backend with GPU acceleration and a modern Next.js frontend with real-time algorithm visualizations.

## 🧭 Approach

- **Goal**: Find strong hyperparameters for DistilBERT using three algorithms (PSO, GA, Bayesian) and compare them side-by-side.
- **Speed vs. Quality**: To iterate fast, optimization runs are executed on a **subset of the training data (20%)**. This drastically reduces turnaround time while preserving relative ranking among configurations.
- **Full-Data Retraining**: After an algorithm finishes, you can retrain a final model on **100% of the data** using the best hyperparameters it found. This provides fair, production-quality metrics.
- **Search Space**: Expanded ranges are used for broader exploration:
  - Learning Rate: `[1e-6, 1e-4]`
  - Batch Size: `[4, 64]`
  - Dropout: `[0.0, 0.5]`
  - Frozen Layers: `[0, 6]`
- **Parameter Selection (Frontend)**:
  - For each hyperparameter, choose to keep it **Optimized** (search) or **Fixed** (set explicit value).
  - Choices persist via the backend in `backend/config/optimization_config.json` and are restored on refresh.
- **Algorithm Runtime Settings**:
  - PSO: `swarmsize`, `maxiter`
  - GA: `population_size`, `num_generations`
  - Bayesian: `n_trials`
  - These settings persist in `backend/logs/settings/algorithm_settings.json`.
- **UI Messaging**: The Training page indicates when results were obtained on **20% of data** and when full-data training is completed on **100%**.

### Key Features

**🔬 Optimization Algorithms**
- **Particle Swarm Optimization (PSO)**: Bio-inspired swarm intelligence
- **Genetic Algorithm (GA)**: Evolutionary computation approach
- **Bayesian Optimization**: Probabilistic model-based optimization
- Side-by-side algorithm comparison with metrics
- Expanded hyperparameter search space (100× learning rate range)

**📊 Real-Time Visualizations**
- Live optimization progress tracking for all algorithms
- Interactive parameter evolution charts (4D visualization)
- Algorithm-specific animations:
  - PSO: Particle swarm movement in hyperparameter space
  - GA: Population evolution and fitness progression
  - Bayesian: Trial history and acquisition function
- Comprehensive metrics dashboard with algorithm comparison

**💻 Modern Stack**
- FastAPI backend with async support
- Next.js 14 with TypeScript
- Tailwind CSS + shadcn/ui components
- Recharts for data visualization
- WebSocket for real-time updates

**🎨 User Experience**
- Responsive design (mobile/tablet/desktop)
- Fullscreen chart viewing
- Dark mode ready
- Interactive model comparison

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- CUDA-capable GPU (optional, for faster training)

### 1. Clone Repository
```bash
git clone https://github.com/i-aissaoui/distilbert-metaheuristic-optimization.git
cd distilbert-metaheuristic-optimization
```

### 2. Backend Setup
```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Train baseline model (15 min on GPU)
python training/train_baseline.py

# Start server
python start_server.py
```

### 3. Frontend Setup
```bash
cd frontend

# Install dependencies
npm install

# Configure environment
cp .env.example .env
echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > .env.local

# Start development server
npm run dev
```

### 4. Access Application
- **Frontend**: http://localhost:3000
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## 🐳 Docker Setup (Recommended)

### Quick Start with Docker
```bash
# Clone repository
git clone https://github.com/i-aissaoui/distilbert-metaheuristic-optimization.git
cd distilbert-metaheuristic-optimization

# Build and run with Docker Compose
docker-compose up -d

# Check logs
docker-compose logs -f

# Access application
# Frontend: http://localhost:3000
# Backend API: http://localhost:8000/docs
```

### Docker Commands
```bash
# Stop containers
docker-compose down

# Rebuild after code changes
docker-compose up --build

# View logs
docker-compose logs backend
docker-compose logs frontend

# Enter container shell
docker-compose exec backend bash
docker-compose exec frontend sh
```

### GPU Support (NVIDIA)
Requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
```bash
# The docker-compose.yml already includes GPU configuration
docker-compose up -d
```

## 📁 Project Structure

```
advanced-ai/
├── backend/                    # Python FastAPI backend
│   ├── app/                   # API application
│   │   ├── main.py           # FastAPI app & endpoints
│   │   ├── config.py         # Configuration
│   │   ├── schemas.py        # Pydantic models
│   │   └── model_loader.py   # Model management
│   ├── training/              # ML training pipeline
│   │   ├── data_loader.py    # Dataset handling
│   │   ├── trainer.py        # Training utilities
│   │   ├── train_baseline.py # Baseline training
│   │   ├── pso_optimization.py # PSO algorithm
│   │   └── train_optimized.py  # Optimized training
│   ├── models/                # Saved models (generated)
│   ├── data/                  # Dataset cache (generated)
│   ├── logs/                  # Training logs & results
│   ├── requirements.txt       # Python dependencies
│   └── Dockerfile            # Container config
│
├── frontend/                  # Next.js frontend
│   ├── app/                  # Next.js app router
│   │   ├── layout.tsx       # Root layout
│   │   ├── page.tsx         # Main page
│   │   └── globals.css      # Global styles
│   ├── components/           # React components
│   │   ├── ui/              # Reusable UI
│   │   ├── PredictionForm.tsx
│   │   ├── PredictionResult.tsx
│   │   └── ModelComparison.tsx
│   ├── lib/                 # Utilities
│   │   ├── api.ts          # API client
│   │   └── utils.ts        # Helpers
│   ├── package.json         # Dependencies
│   ├── tsconfig.json        # TypeScript config
│   └── tailwind.config.ts   # Tailwind config
│
├── docker-compose.yml         # Multi-container setup
├── .gitignore                # Git ignore rules
├── LICENSE                   # MIT License
└── README.md                 # This file
```

## 🛠️ Technologies

### Backend
- **Python 3.10+**: Core language
- **FastAPI**: Modern async web framework
- **PyTorch**: Deep learning framework
- **Transformers**: HuggingFace models
- **pyswarm**: PSO implementation
- **scikit-learn**: Metrics & evaluation

### Frontend
- **Next.js 14**: React framework with App Router
- **TypeScript**: Type-safe development
- **Tailwind CSS**: Utility-first styling
- **Recharts**: Data visualization
- **Lucide React**: Modern icons
- **Axios**: HTTP client

### ML/AI
- **Model**: DistilBERT (66M parameters)
- **Dataset**: TweetEval Hate Speech (~12,000 tweets)
- **Classes**: 2 (Hate / Not Hate)
- **Optimization**: PSO, GA, Bayesian

## 📊 Classification Labels

| Label | Icon | Description |
|-------|------|-------------|
| **hate** | 🚫 | Hate speech, harassment, or toxic targeting |
| **not-hate** | ✅ | Normal, non-offensive message |

## 🔧 API Reference

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check & model status |
| `POST` | `/predict` | Classify text statement |
| `GET` | `/model-info` | Model metrics & performance |
| `GET` | `/labels` | Classification labels |

### Training Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/start-training` | Start custom training |
| `POST` | `/start-pso` | Start PSO optimization |
| `GET` | `/training-status` | Get current training status |
| `GET` | `/training-history` | Training history logs |
| `GET` | `/pso-history` | PSO convergence data |
| `GET` | `/pso-animation` | Particle movement data |
| `GET` | `/hyperparameters` | Model hyperparameters |
| `WS` | `/ws/training` | Real-time training updates |

### Example Usage

**Predict Message:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "text": "You people are disgusting and should not exist",
    "use_optimized": true,
    "model_type": "pso"
  }'
```

**Response:**
```json
{
  "text": "You people are disgusting and should not exist",
  "predicted_label": "hate",
  "confidence": 0.92,
  "all_probabilities": {
    "hate": 0.92,
    "not-hate": 0.08
  },
  "model_used": "pso-optimized"
}
```

## 🎓 Training Pipeline

### Option 1: Baseline Model (Quick)
```bash
cd backend
python training/train_baseline.py
```
**Time**: 15 min (GPU) / 60 min (CPU)

**Hyperparameters:**
- Learning Rate: `2e-5`
- Batch Size: `16`
- Dropout: `0.1`
- Frozen Layers: `0`
- Epochs: `3`

### Option 2: Metaheuristic Optimization (Advanced)

**A. Particle Swarm Optimization (PSO)**
```bash
python training/pso_optimization.py --swarmsize 8 --maxiter 15
```
**Time**: 5-8 hours (GPU) / 12-24 hours (CPU)

**B. Genetic Algorithm (GA)**
```bash
python training/ga_optimization.py --population 10 --generations 15
```
**Time**: 6-9 hours (GPU) / 14-26 hours (CPU)

**C. Bayesian Optimization**
```bash
python training/bayesian_optimization.py --trials 20
```
**Time**: 4-6 hours (GPU) / 10-18 hours (CPU)

**Expanded Search Space (All Algorithms):**
- Learning Rate: `[1e-6, 1e-4]` **(100× wider range)**
- Batch Size: `[4, 64]` **(16 options)**
- Dropout: `[0.0, 0.5]` **(full range)**
- Frozen Layers: `[0, 6]` **(7 options)**

**How PSO Works:**
1. Initialize swarm of particles (each = hyperparameter combination)
2. Train model with each particle's parameters
3. Evaluate F1 score on validation set
4. Update particle positions toward best solutions
5. Repeat until convergence
6. Return optimal hyperparameters

### Option 3: Use UI Training Tab
1. Navigate to http://localhost:3000
2. Click "Training" tab
3. Choose optimization method:
   - **Custom Parameters**: Manual hyperparameter selection
   - **PSO**: Particle swarm intelligence
   - **GA**: Genetic evolution
   - **Bayesian**: Probabilistic sampling (most efficient)
4. Configure algorithm parameters
5. Click "Start Optimization"
6. Watch real-time progress and algorithm-specific visualizations

## 📈 Expected Performance

| Metric | Baseline | Optimized | Improvement |
|--------|----------|-----------|-------------|
| **Accuracy** | ~27% | ~31% | +14.8% |
| **F1 Score** | ~26% | ~30% | +15.4% |
| **Precision** | ~27% | ~31% | +14.8% |
| **Recall** | ~27% | ~31% | +14.8% |

*Note: Results vary based on random initialization and dataset splits.*

## 💻 Development

### Backend
```bash
cd backend
source venv/bin/activate

# Clean startup (suppresses TensorFlow warnings)
python start_server.py

# Standard startup (all logs)
uvicorn app.main:app --reload
```

### Frontend
```bash
cd frontend
npm run dev
```

### Environment Variables

**Backend** (optional):
```bash
CUDA_VISIBLE_DEVICES=0        # GPU device selection
TF_CPP_MIN_LOG_LEVEL=3        # Suppress TF warnings
```

**Frontend** (`.env.local`):
```bash
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### Code Structure

**Backend:**
- `app/main.py` - FastAPI application & routes
- `app/model_loader.py` - Model management
- `training/pso_optimization.py` - PSO algorithm
- `training/trainer.py` - Training utilities

**Frontend:**
- `app/page.tsx` - Main application page
- `app/components/Training.tsx` - Training interface
- `app/components/PSOVisualization.tsx` - PSO charts
- `lib/api.ts` - API client

## 🐳 Docker Deployment

```bash
# Build and run
docker-compose up

# Build only
docker-compose build

# Run in background
docker-compose up -d

# Stop
docker-compose down
```

## 🔍 Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| **No models loaded** | Run `python training/train_baseline.py` |
| **CUDA out of memory** | Reduce batch size or use CPU: `unset CUDA_VISIBLE_DEVICES` |
| **Port 8000 in use** | Change port: `uvicorn app.main:app --port 8001` |
| **Cannot connect to API** | Check `.env.local` has correct `NEXT_PUBLIC_API_URL` |
| **Too many startup logs** | Use `python start_server.py` for clean output |
| **Charts not displaying** | Install recharts: `npm install recharts` |
| **Training logs not saving** | Check browser localStorage is enabled |

### Verification

**Check Backend:**
```bash
curl http://localhost:8000/health
# Should return: {"status":"healthy","models_loaded":{...}}
```

**Check Frontend:**
```bash
# Open browser console (F12)
# Should see no CORS errors
# Network tab should show successful API calls
```

### Performance Tips

- **GPU Training**: Use CUDA for 10x faster training
- **Batch Size**: Increase for faster training (if memory allows)
- **PSO Iterations**: Start with 5 iterations for testing
- **Real-time Updates**: Polling interval set to 2 seconds (configurable)

## 📦 Dataset

**TweetEval Hate Speech Dataset**
- **Training**: ~9,000 tweets
- **Validation**: ~1,000 tweets
- **Test**: ~2,000 tweets
- **Source**: Twitter/X social media content
- **Labels**: Binary (Hate / Not Hate)
- **Focus**: Harassment, toxic targeting, offensive content

## ⚡ Performance

### Training Time (GPU)
- Baseline model: 10-15 minutes
- PSO optimization: 5-8 hours
- Optimized model: 15-20 minutes

### Training Time (CPU)
- Baseline model: 30-60 minutes
- PSO optimization: 12-24 hours
- Optimized model: 45-90 minutes

### Inference Speed
- **GPU**: 50-100 predictions/second (10-20ms latency)
- **CPU**: 5-10 predictions/second (100-200ms latency)

## 🎯 Use Cases

1. **Social Media Moderation**: Automatic hate speech detection
2. **Online Safety**: Protecting users from harassment
3. **Content Filtering**: Pre-moderation systems
4. **Research**: Academic studies on online toxicity
5. **Anti-Bullying**: Detecting cyberbullying
6. **Education**: Teaching AI/ML ethics and NLP

## 📊 Application Features

### 1. Analyze Message Tab
- Enter any social media message or text
- Choose between 4 models: Baseline, PSO, GA, or Bayesian
- View probability distribution (Hate / Not Hate)
- See confidence scores with visual indicators
- Example messages provided

### 2. Model Performance Tab
- Compare baseline vs PSO-optimized models
- View detailed metrics (Accuracy, F1, Precision, Recall)
- See improvement percentages
- Interactive comparison charts
- Fullscreen chart viewing

### 3. Training Tab
- **Custom Parameters**: Train with manual hyperparameters
- **PSO Optimization**: Swarm intelligence optimization
- **Genetic Algorithm**: Evolutionary optimization
- **Bayesian Optimization**: Smart probabilistic sampling
- Real-time progress tracking for all algorithms
- Algorithm-specific visualizations:
  - PSO: Particle swarm animation, convergence plots
  - GA: Population evolution, fitness progression
  - Bayesian: Trial history, acquisition function
- Parameter evolution charts (4 parameters tracked)
- Training history with persistent logs
- Comprehensive metrics comparison across all algorithms

## 🎯 Algorithm Visualizations

The Training tab includes comprehensive visualizations for each algorithm:

### PSO Visualization
- **Learning Rate Evolution**: Track all particles' learning rates
- **Batch Size Evolution**: See batch size optimization
- **Dropout Evolution**: Monitor dropout convergence
- **Frozen Layers Evolution**: Observe layer freezing decisions
- **2D Particle Position**: Learning Rate vs Dropout projection
- **Animation Controls**: Play/pause, speed control, iteration slider
- **Metrics Cards**: Best F1, Accuracy, and optimal parameters

### GA Visualization
- **Population Evolution**: Track fitness across generations
- **Best Individual Progress**: Monitor convergence
- **Diversity Metrics**: Population spread over time
- **Parameter Distribution**: Histogram of hyperparameters

### Bayesian Visualization
- **Trial History**: Sequential trial performance
- **Parameter Importance**: Which hyperparameters matter most
- **Optimization Progress**: Convergence to optimal region

### Comparison Charts (Always Visible)
- **All Algorithms Comparison**: Side-by-side metrics (F1, Accuracy, Time)
- **Hyperparameters Comparison**: Optimal values found by each algorithm
- **Fullscreen Mode**: Expand any chart for detailed analysis

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📄 License

MIT License - see [LICENSE](LICENSE) file

## 🙏 Acknowledgments

- [HuggingFace](https://huggingface.co/) - Transformers & LIAR dataset
- [FastAPI](https://fastapi.tiangolo.com/) - Modern Python web framework
- [Next.js](https://nextjs.org/) - React framework
- [Tailwind CSS](https://tailwindcss.com/) - Utility-first CSS
- [Recharts](https://recharts.org/) - Charting library

---

<div align="center">

**Built with ❤️ using state-of-the-art NLP and optimization techniques**

[Report Bug](https://github.com/i-aissaoui/distilbert-metaheuristic-optimization/issues) · [Request Feature](https://github.com/i-aissaoui/distilbert-metaheuristic-optimization/issues)

</div>
