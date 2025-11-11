#!/bin/bash

echo "🚀 Starting DistilBERT Optimization Backend..."
echo ""

cd backend

# Check if venv exists, create if not
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python -m venv venv
    echo "✓ Virtual environment created"
    echo ""
fi

# Activate venv
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Check if dependencies are installed
if [ ! -f "venv/installed.flag" ]; then
    echo "📦 Installing dependencies (first time only)..."
    pip install -r requirements.txt
    touch venv/installed.flag
    echo "✓ Dependencies installed"
    echo ""
fi

echo "✓ Backend ready!"
echo ""
echo "🌐 Starting server..."
echo "   API: http://localhost:8000"
echo "   Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop"
echo ""

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
