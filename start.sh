#!/bin/bash
# ═══════════════════════════════════════════════════════════
# Nagar Alert Hub — Local Setup & Launch Script
# ═══════════════════════════════════════════════════════════

set -e

BLUE='\033[0;34m'
GREEN='\033[0;32m'
AMBER='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}"
echo "  ███╗   ██╗ █████╗  ██████╗  █████╗ ██████╗ "
echo "  ████╗  ██║██╔══██╗██╔════╝ ██╔══██╗██╔══██╗"
echo "  ██╔██╗ ██║███████║██║  ███╗███████║██████╔╝"
echo "  ██║╚██╗██║██╔══██║██║   ██║██╔══██║██╔══██╗"
echo "  ██║ ╚████║██║  ██║╚██████╔╝██║  ██║██║  ██║"
echo "  ╚═╝  ╚═══╝╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝"
echo -e "  ${AMBER}ALERT HUB${NC} — AI-Powered Public Safety System"
echo ""

# Check Python
if ! command -v python3 &>/dev/null; then
  echo -e "${RED}❌ Python3 not found. Please install Python 3.9+${NC}"
  exit 1
fi

# Virtual env
if [ ! -d "venv" ]; then
  echo -e "${AMBER}⚙️  Creating virtual environment...${NC}"
  python3 -m venv venv
fi
source venv/bin/activate

# Install dependencies
echo -e "${AMBER}📦 Installing dependencies...${NC}"
pip install -q -r requirements.txt

# Copy env
if [ ! -f ".env" ]; then
  echo -e "${AMBER}⚙️  Creating .env from template...${NC}"
  cp .env.example .env
  echo -e "${GREEN}✅ .env created — add your GOOGLE_MAPS_API_KEY to enable Google Maps${NC}"
fi

# Train model if not exists
if [ ! -f "models/incident_classifier.pkl" ]; then
  echo -e "${AMBER}🤖 Training ML model (first time only)...${NC}"
  python models/train_model.py
  echo -e "${GREEN}✅ Model trained and saved${NC}"
else
  echo -e "${GREEN}✅ Model already trained${NC}"
fi

# Create DB directory
mkdir -p instance

echo ""
echo -e "${GREEN}🚀 Starting Nagar Alert Hub...${NC}"
echo -e "   ${BLUE}Local:   http://localhost:5000${NC}"
echo -e "   ${BLUE}Network: http://$(hostname -I | awk '{print $1}'):5000${NC}"
echo ""

# Run Flask
FLASK_ENV=development python app.py
