# 📋 Quick Reference Guide

## 🚀 Getting Started (3 Commands)

```bash
# 1. Clone and enter directory
git clone https://github.com/YOUR_USERNAME/ai-fake-news-detector.git
cd ai-fake-news-detector

# 2. Start with Docker
docker-compose up -d

# 3. Train baseline model
docker-compose exec backend python training/train_baseline.py
```

**Access:** http://localhost:3000

---

## 🐳 Docker Commands

```bash
# Start
docker-compose up -d

# Stop
docker-compose down

# Rebuild
docker-compose up --build

# Logs
docker-compose logs -f
docker-compose logs backend
docker-compose logs frontend

# Shell access
docker-compose exec backend bash
docker-compose exec frontend sh

# Restart single service
docker-compose restart backend
```

---

## 🎯 Training Commands

### Inside Backend Container
```bash
# Enter container
docker-compose exec backend bash

# Baseline (15 min GPU / 60 min CPU)
python training/train_baseline.py

# PSO Optimization (5-8 hours GPU)
python training/pso_optimization.py 8 15

# GA Optimization (6-9 hours GPU)
python training/ga_optimization.py 10 15

# Bayesian Optimization (4-6 hours GPU)
python training/bayesian_optimization.py 20
```

---

## 🔧 API Endpoints

### Core
```bash
# Health check
curl http://localhost:8000/health

# Predict
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"Your statement here","use_optimized":false}'

# Model info
curl http://localhost:8000/model-info
```

### Training
```bash
# Start PSO
curl -X POST http://localhost:8000/start-pso?swarmsize=8&maxiter=15

# Start GA
curl -X POST http://localhost:8000/start-ga?population_size=10&num_generations=15

# Start Bayesian
curl -X POST http://localhost:8000/start-bayesian?n_trials=20

# Training status
curl http://localhost:8000/training-status

# PSO history
curl http://localhost:8000/pso-history

# GA history
curl http://localhost:8000/ga-history

# Bayesian history
curl http://localhost:8000/bayesian-history
```

---

## 📂 Important Files

### Configuration
- `docker-compose.yml` - Container orchestration
- `backend/requirements.txt` - Python dependencies
- `frontend/package.json` - Node dependencies
- `.gitignore` - Git ignore rules
- `.env.local` - Frontend environment variables

### Models & Data
- `backend/models/` - Trained models (generated)
- `backend/data/` - Dataset cache (generated)
- `backend/logs/results/` - Training results (generated)

### Code
- `backend/app/main.py` - FastAPI application
- `backend/training/pso_optimization.py` - PSO algorithm
- `backend/training/ga_optimization.py` - GA algorithm
- `backend/training/bayesian_optimization.py` - Bayesian algorithm
- `frontend/app/page.tsx` - Main application page
- `frontend/app/components/Training.tsx` - Training interface

---

## 🐛 Common Issues & Fixes

### Models not loaded
```bash
docker-compose exec backend python training/train_baseline.py
```

### Port 8000 in use
```bash
# Edit docker-compose.yml, change ports:
ports:
  - "8001:8000"
```

### CUDA out of memory
```bash
# Remove GPU section from docker-compose.yml or reduce batch size
```

### Frontend can't connect to backend
```bash
# Check .env.local
echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > frontend/.env.local
docker-compose restart frontend
```

### Clear all data and restart
```bash
docker-compose down -v
rm -rf backend/models/* backend/data/* backend/logs/results/*
docker-compose up -d
docker-compose exec backend python training/train_baseline.py
```

---

## 📊 Expected Results

### Baseline Model
- **Training Time:** 15 min (GPU) / 60 min (CPU)
- **Accuracy:** ~27%
- **F1 Score:** ~26%

### Optimized Model (PSO/GA/Bayesian)
- **Training Time:** 5-8 hours (GPU) / 12-24 hours (CPU)
- **Accuracy:** ~31% (+14.8%)
- **F1 Score:** ~30% (+15.4%)

---

## 🎨 UI Features

### Analyze Statement Tab
- Enter text to classify
- Choose baseline or optimized model
- View probability distribution
- See confidence scores

### Model Performance Tab
- Compare baseline vs optimized
- View detailed metrics
- Interactive charts
- Fullscreen mode

### Training Tab
- Custom parameter training
- PSO optimization with swarm visualization
- GA optimization with evolution tracking
- Bayesian optimization with trial history
- Real-time progress tracking
- Algorithm comparison charts

---

## 🔐 Security (Production)

### Add API Authentication
```python
# backend/app/main.py
from fastapi import Security, HTTPException
from fastapi.security import APIKeyHeader

API_KEY = "your-secret-key-here"
api_key_header = APIKeyHeader(name="X-API-Key")

async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403)
    return api_key

@app.post("/predict", dependencies=[Depends(verify_api_key)])
async def predict(...):
    ...
```

### Enable HTTPS
```yaml
# docker-compose.yml
services:
  nginx:
    image: nginx:alpine
    ports:
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
```

---

## 📈 Monitoring

### Check Logs
```bash
# Real-time logs
docker-compose logs -f

# Last 100 lines
docker-compose logs --tail=100

# Specific service
docker-compose logs backend --tail=50
```

### Check Resource Usage
```bash
# Container stats
docker stats

# Disk usage
docker system df
```

### Health Checks
```bash
# Backend
curl http://localhost:8000/health

# Frontend
curl http://localhost:3000

# API docs
open http://localhost:8000/docs
```

---

## 🔄 Update & Maintenance

### Update Code
```bash
git pull origin main
docker-compose up --build -d
```

### Backup Models
```bash
tar -czf backup_$(date +%Y%m%d).tar.gz backend/models/ backend/logs/
```

### Clean Docker
```bash
# Remove unused containers
docker system prune

# Remove all volumes (WARNING: deletes data)
docker-compose down -v
```

---

## 📞 Support

- **Documentation:** README.md
- **Deployment:** DEPLOYMENT_GUIDE.md
- **Issues:** https://github.com/YOUR_USERNAME/ai-fake-news-detector/issues
- **API Docs:** http://localhost:8000/docs (when running)

---

**Quick Tip:** Bookmark this file for instant reference! 🔖
