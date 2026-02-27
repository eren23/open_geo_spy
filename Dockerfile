FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0t64 \
    curl \
    chromium \
    chromium-driver \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# Create playwright directory and set permissions
RUN mkdir -p /ms-playwright && \
    chown -R root:root /ms-playwright && \
    chmod -R 777 /ms-playwright

ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

# Copy source and install Python dependencies
COPY pyproject.toml .
COPY src/ ./src/
RUN pip install --no-cache-dir .

RUN PLAYWRIGHT_BROWSERS_PATH=/ms-playwright playwright install chromium --with-deps || true
COPY .env .

# Set Python path and environment variables
ENV PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1

# Set environment variable for Chrome
ENV CHROME_BIN=/usr/bin/chromium

# Startup script
RUN echo '#!/bin/bash\n\
echo "Starting Xvfb..."\n\
Xvfb :99 -screen 0 1024x768x16 &\n\
export DISPLAY=:99\n\
echo "Starting OpenGeoSpy API..."\n\
exec uvicorn src.api.app:app --host 0.0.0.0 --port 8000' > /app/start.sh && \
    chmod +x /app/start.sh

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

CMD ["/app/start.sh"]
