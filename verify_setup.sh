#!/bin/bash

# Verification script to check if everything is set up correctly

echo "🔍 Verifying Fake News Detection System Setup..."
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check counters
PASS=0
FAIL=0

# Function to check
check() {
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓${NC} $1"
        ((PASS++))
    else
        echo -e "${RED}✗${NC} $1"
        ((FAIL++))
    fi
}

echo "📁 Checking Project Structure..."
echo "================================"

# Check backend structure
[ -d "backend" ] && check "Backend directory exists" || check "Backend directory exists"
[ -d "backend/app" ] && check "Backend app directory exists" || check "Backend app directory exists"
[ -d "backend/training" ] && check "Backend training directory exists" || check "Backend training directory exists"
[ -d "backend/models" ] && check "Backend models directory exists" || check "Backend models directory exists"
[ -d "backend/logs" ] && check "Backend logs directory exists" || check "Backend logs directory exists"

# Check frontend structure
[ -d "frontend" ] && check "Frontend directory exists" || check "Frontend directory exists"
[ -d "frontend/app" ] && check "Frontend app directory exists" || check "Frontend app directory exists"
[ -d "frontend/components" ] && check "Frontend components directory exists" || check "Frontend components directory exists"
[ -d "frontend/lib" ] && check "Frontend lib directory exists" || check "Frontend lib directory exists"

echo ""
echo "📄 Checking Key Files..."
echo "========================"

# Backend files
[ -f "backend/app/main.py" ] && check "FastAPI main.py exists" || check "FastAPI main.py exists"
[ -f "backend/requirements.txt" ] && check "Backend requirements.txt exists" || check "Backend requirements.txt exists"
[ -f "backend/models/performance_comparison.json" ] && check "Demo performance data exists" || check "Demo performance data exists"
[ -f "backend/logs/training_history.json" ] && check "Demo training history exists" || check "Demo training history exists"
[ -f "backend/logs/pso_animation.json" ] && check "Demo PSO animation exists" || check "Demo PSO animation exists"

# Frontend files
[ -f "frontend/package.json" ] && check "Frontend package.json exists" || check "Frontend package.json exists"
[ -f "frontend/app/page.tsx" ] && check "Main page exists" || check "Main page exists"
[ -f "frontend/app/training/page.tsx" ] && check "Training page exists" || check "Training page exists"
[ -f "frontend/app/testing/page.tsx" ] && check "Testing page exists" || check "Testing page exists"

echo ""
echo "🔧 Checking Dependencies..."
echo "==========================="

# Check Python
if command -v python3 &> /dev/null; then
    check "Python 3 is installed"
else
    check "Python 3 is installed"
fi

# Check Node
if command -v node &> /dev/null; then
    check "Node.js is installed"
    NODE_VERSION=$(node --version)
    echo "   Node version: $NODE_VERSION"
else
    check "Node.js is installed"
fi

# Check npm
if command -v npm &> /dev/null; then
    check "npm is installed"
else
    check "npm is installed"
fi

echo ""
echo "📦 Checking Installations..."
echo "============================"

# Check if frontend node_modules exists
if [ -d "frontend/node_modules" ]; then
    check "Frontend dependencies installed"
else
    echo -e "${YELLOW}⚠${NC}  Frontend dependencies not installed"
    echo "   Run: cd frontend && npm install"
    ((FAIL++))
fi

# Check if backend venv exists
if [ -d "backend/venv" ]; then
    check "Backend virtual environment exists"
else
    echo -e "${YELLOW}⚠${NC}  Backend virtual environment not found"
    echo "   Run: cd backend && python -m venv venv"
    ((FAIL++))
fi

echo ""
echo "🎯 Summary"
echo "=========="
echo -e "${GREEN}Passed: $PASS${NC}"
echo -e "${RED}Failed: $FAIL${NC}"

echo ""
if [ $FAIL -eq 0 ]; then
    echo -e "${GREEN}✓ All checks passed! Your project is ready to run.${NC}"
    echo ""
    echo "🚀 Next Steps:"
    echo "1. Start backend: cd backend && uvicorn app.main:app --reload"
    echo "2. Frontend should already be running at http://localhost:3000"
    echo "3. Check START_HERE.md for detailed instructions"
else
    echo -e "${YELLOW}⚠ Some checks failed. Please review the issues above.${NC}"
    echo ""
    echo "📖 See START_HERE.md for setup instructions"
fi

echo ""
