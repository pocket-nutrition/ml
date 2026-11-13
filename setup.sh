#!/usr/bin/env bash
set -euo pipefail

# pocket-nutrition-ml — First-time setup
# Usage: ./setup.sh

echo "=== pocket-nutrition-ml Setup ==="

# Check prerequisites
command -v python3.12 >/dev/null 2>&1 || { echo "Error: python3.12 is required."; exit 1; }

# Virtual environment
if [ ! -d .venv ]; then
  echo "Creating virtual environment..."
  python3.12 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

# Environment
if [ ! -f .env ]; then
  cp .env.example .env
  echo "Created .env from .env.example — edit it with your values"
fi

# Dependencies
echo "Installing dependencies..."
pip install --upgrade pip >/dev/null
pip install -r requirements.txt pytest

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Next steps:"
echo "  1. Edit .env with your configuration (MODEL_PATH; a model artifact is optional — /predict"
echo "     still runs and returns result: null without one)"
echo "  2. Activate the virtual environment: source .venv/bin/activate"
echo "  3. Run: uvicorn app.main:app --reload --port 8000"
echo "  4. Open: http://localhost:8000/health"
echo "  5. Run tests: pytest tests/ -v"
echo "  6. Using Claude Code? CLAUDE.md has all the context."
