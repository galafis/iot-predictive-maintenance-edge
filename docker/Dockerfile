FROM python:3.11-slim

LABEL maintainer="Gabriel Demetrios Lafis"
LABEL description="IoT Predictive Maintenance Edge Computing Platform"

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ src/
COPY config/ config/
COPY main.py .

# Create directories for models and logs
RUN mkdir -p models/ logs/

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV APP_ENV=production
ENV LOG_LEVEL=INFO

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import src; print('healthy')" || exit 1

# Run the application
CMD ["python", "main.py"]
