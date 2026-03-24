#!/usr/bin/env bash
set -e
echo "▶ Installing Python dependencies..."
pip install -r requirements.txt

echo "▶ Downloading sentence-transformers model..."
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

echo "▶ Checking Ollama..."
if ! command -v ollama &>/dev/null; then
  echo "  Ollama not found. Install from https://ollama.ai then run: ollama pull qwen2.5:0.5b"
else
  ollama pull "${MODEL_NAME:-qwen2.5:0.5b}"
fi

echo "✓ Setup complete. Run: bash scripts/run.sh"
