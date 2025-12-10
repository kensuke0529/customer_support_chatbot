# =============================================================================
# Customer Support AI Agent - Production Dockerfile
# =============================================================================
# This Dockerfile addresses:
# - Health check failures (0.0.0.0 binding)
# - Architecture compatibility (Apple Silicon / AWS)
# - Security (non-root user)
# - CloudWatch logging (unbuffered output)
# =============================================================================

# 1. Base Image: Use specific, slim version for reproducibility and minimal size
FROM python:3.11-slim

# 2. Environment Variables
# PYTHONDONTWRITEBYTECODE: Prevents creation of .pyc files (cleaner container)
# PYTHONUNBUFFERED: ESSENTIAL for CloudWatch logs - without this, logs are buffered and may be lost on crash
# PORT: Default to 8080 to match AWS App Runner standard
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

# 3. System Dependencies
# Install curl for health check testing and build-essential for any native deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4. Application Directory
WORKDIR /app

# 5. Install Python Dependencies
# Copy requirements first for better Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy Application Code
COPY . .

# 7. Security: Run as non-root user
# Creating a dedicated user prevents potential container breakout vulnerabilities
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# 8. Expose port (documentation - App Runner uses PORT env var)
EXPOSE 8080

# 9. Health Check (optional - App Runner has its own, but useful for local Docker)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# 10. Entrypoint
# - Binds to 0.0.0.0 to fix App Runner health check failures
# - Uses explicit port 8080 (App Runner standard) for reliability
# - Single worker for predictable memory usage on App Runner
CMD uvicorn backend.app.main:app --host 0.0.0.0 --port 8080 --workers 1
