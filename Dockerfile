FROM python:3.13-slim

# Set work directory
WORKDIR /app

# Ensure no interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set Hugging Face cache to a writeable location
ENV HF_HOME=/app/hf_cache
ENV TRANSFORMERS_CACHE=/app/hf_cache

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir uvicorn sentence-transformers

# Pre-download the model to cache
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('LaBSE')"

# Copy application code
COPY . .

# Expose port for Hugging Face
EXPOSE 7860

# Run as non-root user (preferred on HF Spaces)
RUN useradd -m appuser
USER appuser

# Start app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
