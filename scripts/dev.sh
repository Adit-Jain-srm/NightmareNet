#!/usr/bin/env bash
# Enable strict error tracking (Bot requirement)
set -o pipefail

# Colors defined for safe injection (Bot requirement)
BLUE=$'\033[0;34m'
YELLOW=$'\033[0;33m'
RED=$'\033[0;31m'
NC=$'\033[0m'

API_PID=""
FE_PID=""

# Advanced Graceful Cleanup (Bot requirement)
cleanup() {
    echo -e "\n${RED}[System] Shutting down servers gracefully...${NC}"
    if [ -n "$API_PID" ]; then kill -TERM "$API_PID" 2>/dev/null || true; fi
    if [ -n "$FE_PID" ]; then kill -TERM "$FE_PID" 2>/dev/null || true; fi
    exit 1
}

# Trap both SIGINT (Ctrl+C) and SIGTERM (Bot requirement)
trap cleanup SIGINT SIGTERM

# --- Boot API (FastAPI) ---
echo -e "${BLUE}[API] Starting Uvicorn development server...${NC}"
uvicorn nightmarenet.api.app:app --reload --host 127.0.0.1 --port 8000 2>&1 | sed -u -e "s/^/${BLUE}[API]${NC} /" &
API_PID=$!

# --- Boot Frontend (Next.js) ---
echo -e "${YELLOW}[Frontend] Starting Next.js development server...${NC}"
(cd frontend && npm run dev) 2>&1 | sed -u -e "s/^/${YELLOW}[Frontend]${NC} /" &
FE_PID=$!

# Wait for any process to crash/exit (Bot requirement)
wait -n

# If one crashes, it reaches here and cleans up the other
cleanup
