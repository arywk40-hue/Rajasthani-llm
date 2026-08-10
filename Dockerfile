FROM python:3.10-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user and group
RUN groupadd -g 10001 appuser && \
    useradd -u 10001 -g appuser -d /app -s /bin/bash appuser

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and set ownership
COPY --chown=appuser:appuser . .
RUN pip install -e .

# Make app directory writable for Hugging Face caching
RUN mkdir -p /app/.cache && chown -R appuser:appuser /app

# Set non-root execution context
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health')" || exit 1

# Run with uvicorn
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
