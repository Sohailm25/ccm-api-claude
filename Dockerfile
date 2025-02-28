FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for sentence-transformers
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev \
    libjpeg-dev \
    zlib1g-dev \
    libxml2-dev \
    libxslt1-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Fix lxml and newspaper3k compatibility issues
RUN pip uninstall -y lxml && pip install "lxml[html_clean]==4.9.3"
RUN pip install --no-cache-dir newspaper3k==0.2.8
RUN pip install --no-cache-dir youtube-transcript-api==0.6.1

# Pre-download sentence transformers model to avoid timeout during startup
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-MiniLM-L3-v2')"

# Copy application code
COPY . .

# Add trafilatura installation
RUN pip install --no-cache-dir trafilatura

# After installing requirements
RUN pip install --no-cache-dir anthropic==0.8.1

# Run the application
CMD python -c "from database import create_tables; create_tables()" && uvicorn main:app --host 0.0.0.0 --port $PORT 