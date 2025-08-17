FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY xtts_server.py .
COPY dist/reference_kelly.wav dist/
COPY dist/reference_ken_mono16k.wav dist/reference_ken_mono16k.wav
# Create a symlink for consistency
RUN ln -s /app/dist/reference_ken_mono16k.wav /app/dist/reference_ken.wav

# Pre-download the model to speed up cold starts
RUN python -c "from TTS.api import TTS; TTS('tts_models/multilingual/multi-dataset/xtts_v2')"

# Environment
ENV PYTHONUNBUFFERED=1
ENV PORT=5002

# Expose port
EXPOSE 5002

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD python -c "import requests; exit(0 if requests.get('http://localhost:5002/health').status_code == 200 else 1)"

# Run server
CMD ["python", "xtts_server.py"]
