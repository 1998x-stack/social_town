FROM python:3.11-slim

WORKDIR /app

# Install curl (needed for healthcheck polling)
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Set HF_HOME before pre-downloading so model is cached to the correct location
ENV HF_HOME=/app/.cache/huggingface

# Pre-download sentence-transformers model into image layer
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY . .

ENV PYTHONUNBUFFERED=1
EXPOSE 8080

CMD ["python", "main.py"]
