#!/bin/sh
set -e
MODEL_NAME="${MODEL_NAME:-qwen2.5:0.5b}"

# Start Ollama serve in background
ollama serve &
SERVE_PID=$!

# Wait for API readiness
echo "[ollama-init] Waiting for Ollama API..."
MAX_RETRIES=30
RETRIES=0
until curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; do
  RETRIES=$((RETRIES + 1))
  if [ $RETRIES -ge $MAX_RETRIES ]; then
    echo "ERROR: Ollama failed to start after $MAX_RETRIES retries"
    exit 1
  fi
  sleep 2
done
echo "[ollama-init] Ollama API ready"

# Pull model if not present
echo "[ollama-init] Pulling model: $MODEL_NAME"
ollama pull "$MODEL_NAME"
echo "[ollama-init] Model ready: $MODEL_NAME"

# Keep Ollama running in foreground
wait $SERVE_PID
