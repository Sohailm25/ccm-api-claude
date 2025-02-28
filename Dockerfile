FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for sentence-transformers
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# We don't need to install these packages separately since they're in requirements.txt
# Removed duplicate installations

# Pre-download sentence transformers model to avoid timeout during startup
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-MiniLM-L3-v2')"

# Copy application code
COPY . .

# Add memory limit to prevent OOM kills
ENV PYTHONUNBUFFERED=1
ENV PYTHONMALLOC=malloc
ENV MALLOC_TRIM_THRESHOLD_=65536

# Run the application with increased memory efficiency
CMD python -c "from database import create_tables; create_tables()" && uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1 --limit-concurrency 10 