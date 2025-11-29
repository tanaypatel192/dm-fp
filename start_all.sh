#!/bin/bash
# Shell script to start both backend and frontend
# Usage: ./start_all.sh

echo ""
echo "========================================"
echo "  Diabetes Prediction System Launcher"
echo "========================================"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}ERROR: Python 3 is not installed or not in PATH${NC}"
    exit 1
fi

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo -e "${RED}ERROR: Node.js is not installed or not in PATH${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Python found: $(python3 --version)${NC}"
echo -e "${GREEN}✓ Node.js found: $(node --version)${NC}"
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo -e "${YELLOW}Stopping servers...${NC}"
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    echo -e "${GREEN}Servers stopped.${NC}"
    exit 0
}

# Set trap to cleanup on Ctrl+C
trap cleanup SIGINT SIGTERM

# Start backend
echo -e "${CYAN}[1/2] Starting Backend Server...${NC}"
cd backend

# Activate virtual environment if it exists
if [ -f "venv/bin/activate" ]; then
    echo "      Activating virtual environment..."
    source venv/bin/activate
fi

# Start backend in background
python3 app.py > ../backend.log 2>&1 &
BACKEND_PID=$!

cd ..

# Wait for backend to start
echo "      Waiting for backend to initialize..."
sleep 5

# Check if backend is running
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo -e "      ${GREEN}✓ Backend is healthy!${NC}"
else
    echo -e "      ${YELLOW}⚠ Backend health check failed, but continuing...${NC}"
fi

# Start frontend
echo ""
echo -e "${CYAN}[2/2] Starting Frontend Server...${NC}"
cd frontend

# Start frontend in background
npm run dev > ../frontend.log 2>&1 &
FRONTEND_PID=$!

cd ..

# Wait for frontend to start
echo "      Waiting for frontend to initialize..."
sleep 5

# Check if frontend is running
if curl -s http://localhost:5173 > /dev/null 2>&1; then
    echo -e "      ${GREEN}✓ Frontend is accessible!${NC}"
else
    echo -e "      ${YELLOW}⚠ Frontend check failed, but it may still be starting...${NC}"
fi

echo ""
echo -e "${GREEN}========================================"
echo "  SYSTEM STARTED SUCCESSFULLY!"
echo "========================================${NC}"
echo ""

echo "Application URLs:"
echo -e "  Frontend:     ${CYAN}http://localhost:5173${NC}"
echo -e "  Backend API:  ${CYAN}http://localhost:8000/docs${NC}"
echo -e "  Health Check: ${CYAN}http://localhost:8000/health${NC}"

echo ""
echo "View logs:"
echo "  Backend:  tail -f backend.log"
echo "  Frontend: tail -f frontend.log"

echo ""
echo -e "Press ${YELLOW}Ctrl+C${NC} to stop all servers."

# Run quick tests
echo ""
echo -e "${CYAN}========================================"
echo "  Running Quick Tests..."
echo "========================================${NC}"
echo ""

sleep 3
python3 quick_test.py

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ All tests passed!${NC}"
else
    echo ""
    echo -e "${YELLOW}⚠ Some tests failed. Check the output above.${NC}"
fi

# Open browser (try different commands based on OS)
echo ""
echo "Opening browser..."
if command -v xdg-open &> /dev/null; then
    xdg-open http://localhost:5173
elif command -v open &> /dev/null; then
    open http://localhost:5173
fi

echo ""
echo -e "${GREEN}Setup complete! Happy testing! 🚀${NC}"
echo ""

# Keep script running
wait




