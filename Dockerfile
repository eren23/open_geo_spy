FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
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

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN PLAYWRIGHT_BROWSERS_PATH=/ms-playwright playwright install chromium --with-deps

# Copy application code
COPY src/ ./src/
COPY .env .

# Set Python path and environment variables
ENV PYTHONPATH=/app/src \
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
echo "Starting uvicorn..."\n\
exec uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload' > /app/start.sh && \
    chmod +x /app/start.sh

# Expose the API port
EXPOSE 8000
# Run the startup script
CMD ["/app/start.sh"] 