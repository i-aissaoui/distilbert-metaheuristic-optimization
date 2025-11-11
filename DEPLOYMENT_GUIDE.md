# 🚀 Deployment Guide

## GitHub Repository Setup

### Suggested Repository Name
**`ai-fake-news-detector`** or **`distilbert-metaheuristic-optimizer`**

### Repository Description
```
AI-powered fake news detection using DistilBERT optimized with PSO, GA, and Bayesian algorithms. Full-stack app with FastAPI backend and Next.js frontend.
```

### Topics/Tags
```
machine-learning, nlp, distilbert, particle-swarm-optimization, genetic-algorithm, bayesian-optimization, fastapi, nextjs, pytorch, transformers, fake-news-detection, deep-learning, metaheuristics
```

---

## Pre-Push Checklist

### ✅ Files Cleaned
- [x] Removed `.sh` scripts (quick_start.sh, run_all_training.sh, start_clean.sh)
- [x] Removed redundant README files (backend/logs/results/README.md, backend/models/README.md)
- [x] Removed IMPLEMENTATION_SUMMARY.md
- [x] Updated .gitignore to exclude logs and temp files

### ✅ Docker Support Added
- [x] Dockerfile.backend created
- [x] Dockerfile.frontend created  
- [x] docker-compose.yml configured with GPU support
- [x] README updated with Docker instructions

### ✅ Documentation Complete
- [x] Comprehensive README.md with all features
- [x] API documentation included
- [x] Training pipeline explained
- [x] Troubleshooting guide added

---

## Push to GitHub

### 1. Initialize Git (if not already done)
```bash
cd "/home/ismail/Desktop/projects/ai/advanced ai"
git init
git add .
git commit -m "Initial commit: AI Fake News Detector with metaheuristic optimization"
```

### 2. Create GitHub Repository
1. Go to https://github.com/new
2. Repository name: `ai-fake-news-detector`
3. Description: "AI-powered fake news detection using DistilBERT optimized with PSO, GA, and Bayesian algorithms"
4. Public or Private (your choice)
5. **Do NOT** initialize with README (we already have one)
6. Click "Create repository"

### 3. Push to GitHub
```bash
# Add remote
git remote add origin https://github.com/YOUR_USERNAME/ai-fake-news-detector.git

# Push to main branch
git branch -M main
git push -u origin main
```

---

## Docker Deployment

### Local Docker Setup
```bash
# Build and run
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

### Production Deployment Options

#### Option 1: AWS ECS/Fargate
```bash
# Build images
docker build -f Dockerfile.backend -t ai-detector-backend:latest ./backend
docker build -f Dockerfile.frontend -t ai-detector-frontend:latest ./frontend

# Tag for ECR
docker tag ai-detector-backend:latest YOUR_ECR_URL/ai-detector-backend:latest
docker tag ai-detector-frontend:latest YOUR_ECR_URL/ai-detector-frontend:latest

# Push to ECR
docker push YOUR_ECR_URL/ai-detector-backend:latest
docker push YOUR_ECR_URL/ai-detector-frontend:latest
```

#### Option 2: Google Cloud Run
```bash
# Build and push backend
gcloud builds submit --tag gcr.io/PROJECT_ID/ai-detector-backend backend/

# Build and push frontend
gcloud builds submit --tag gcr.io/PROJECT_ID/ai-detector-frontend frontend/

# Deploy
gcloud run deploy ai-detector-backend --image gcr.io/PROJECT_ID/ai-detector-backend
gcloud run deploy ai-detector-frontend --image gcr.io/PROJECT_ID/ai-detector-frontend
```

#### Option 3: DigitalOcean App Platform
1. Connect GitHub repository
2. Select `docker-compose.yml`
3. Configure environment variables
4. Deploy

---

## Environment Variables

### Backend (.env)
```bash
# Optional - GPU device selection
CUDA_VISIBLE_DEVICES=0

# Optional - Suppress TensorFlow warnings
TF_CPP_MIN_LOG_LEVEL=3
```

### Frontend (.env.local)
```bash
# Required - Backend API URL
NEXT_PUBLIC_API_URL=http://localhost:8000

# Production
NEXT_PUBLIC_API_URL=https://your-backend-url.com
```

---

## Post-Deployment

### 1. Train Baseline Model
```bash
# SSH into backend container or server
docker-compose exec backend bash

# Train baseline
python training/train_baseline.py
```

### 2. Verify Deployment
```bash
# Check backend health
curl https://your-backend-url.com/health

# Should return:
# {"status":"healthy","models_loaded":{"baseline":true,"optimized":false}}
```

### 3. Run Optimization (Optional)
```bash
# PSO optimization (5-8 hours on GPU)
python training/pso_optimization.py 8 15

# GA optimization (6-9 hours on GPU)
python training/ga_optimization.py 10 15

# Bayesian optimization (4-6 hours on GPU)
python training/bayesian_optimization.py 20
```

---

## Monitoring

### Docker Logs
```bash
# All services
docker-compose logs -f

# Backend only
docker-compose logs -f backend

# Frontend only
docker-compose logs -f frontend
```

### Health Checks
```bash
# Backend health
curl http://localhost:8000/health

# Frontend (should return HTML)
curl http://localhost:3000
```

---

## Troubleshooting

### Issue: Models not loading
**Solution:** Train baseline model first
```bash
docker-compose exec backend python training/train_baseline.py
```

### Issue: CUDA out of memory
**Solution:** Reduce batch size or use CPU
```bash
# Edit docker-compose.yml, remove GPU section
# Or reduce batch size in training scripts
```

### Issue: Port conflicts
**Solution:** Change ports in docker-compose.yml
```yaml
services:
  backend:
    ports:
      - "8001:8000"  # Change 8000 to 8001
```

---

## Security Considerations

### Production Checklist
- [ ] Change default ports
- [ ] Add authentication/API keys
- [ ] Enable HTTPS/SSL
- [ ] Set up CORS properly
- [ ] Add rate limiting
- [ ] Monitor logs for suspicious activity
- [ ] Regular security updates

### Recommended: Add API Authentication
```python
# backend/app/main.py
from fastapi import Security, HTTPException
from fastapi.security import APIKeyHeader

API_KEY = "your-secret-key"
api_key_header = APIKeyHeader(name="X-API-Key")

async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key

# Add to endpoints
@app.post("/predict", dependencies=[Depends(verify_api_key)])
async def predict(...):
    ...
```

---

## Maintenance

### Update Dependencies
```bash
# Backend
cd backend
pip install --upgrade -r requirements.txt
pip freeze > requirements.txt

# Frontend
cd frontend
npm update
npm audit fix
```

### Backup Models
```bash
# Backup trained models
tar -czf models_backup_$(date +%Y%m%d).tar.gz backend/models/

# Backup training results
tar -czf results_backup_$(date +%Y%m%d).tar.gz backend/logs/results/
```

---

## Performance Optimization

### Backend
- Use Gunicorn with multiple workers
- Enable Redis caching for predictions
- Batch predictions when possible
- Use model quantization for faster inference

### Frontend
- Enable Next.js static generation
- Use CDN for assets
- Implement lazy loading
- Add service worker for offline support

---

## Support

For issues, questions, or contributions:
- 📧 Email: your-email@example.com
- 🐛 Issues: https://github.com/YOUR_USERNAME/ai-fake-news-detector/issues
- 💬 Discussions: https://github.com/YOUR_USERNAME/ai-fake-news-detector/discussions

---

**Built with ❤️ using state-of-the-art NLP and optimization techniques**
