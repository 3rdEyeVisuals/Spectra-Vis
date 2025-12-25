#!/bin/bash
# ============================================================================
# Spectra Vis - Full Stack Launcher (Linux/macOS)
# Copyright (c) 2025 3rdEyeVisuals
# ============================================================================

echo "============================================================"
echo "  Spectra Vis - Tensor Visualization"
echo "  Copyright (c) 2025 3rdEyeVisuals"
echo "============================================================"
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    if ! command -v python &> /dev/null; then
        echo "ERROR: Python is not installed or not in PATH."
        echo "Please install Python 3.8+ and try again."
        exit 1
    fi
    PYTHON_CMD="python"
else
    PYTHON_CMD="python3"
fi

# Show Python version
$PYTHON_CMD --version

# Check if Node.js is available
if ! command -v node &> /dev/null; then
    echo "ERROR: Node.js is not installed or not in PATH."
    echo "Please install Node.js 18+ and try again."
    exit 1
fi

# Show Node version
echo "Node.js version: $(node --version)"

# Check if backend dependencies are installed
echo ""
echo "Checking backend dependencies..."
if ! $PYTHON_CMD -c "import fastapi, uvicorn" 2>/dev/null; then
    echo "Installing backend dependencies..."
    pip3 install -r backend/requirements.txt || pip install -r backend/requirements.txt
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to install backend dependencies."
        exit 1
    fi
fi

# Check if frontend dependencies are installed
echo "Checking frontend dependencies..."
if [ ! -d "frontend/node_modules" ]; then
    echo "Installing frontend dependencies..."
    cd frontend
    npm install
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to install frontend dependencies."
        exit 1
    fi
    cd ..
fi

echo ""
echo "============================================================"
echo "Starting Spectra Vis..."
echo ""
echo "  Backend API:  http://localhost:8000"
echo "  Frontend UI:  http://localhost:3000"
echo "  API Docs:     http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop both servers."
echo "============================================================"
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "Shutting down servers..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    exit 0
}

trap cleanup SIGINT SIGTERM

# Start backend in background
$PYTHON_CMD backend/server.py &
BACKEND_PID=$!

# Wait for backend to start
sleep 2

# Start frontend in background
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

# Wait for frontend to start
sleep 3

# Open browser
if command -v open &> /dev/null; then
    # macOS
    open http://localhost:3000
elif command -v xdg-open &> /dev/null; then
    # Linux
    xdg-open http://localhost:3000
fi

echo "Both servers started. Browser opening..."
echo "Press Ctrl+C to stop."

# Wait for both processes
wait $BACKEND_PID $FRONTEND_PID
