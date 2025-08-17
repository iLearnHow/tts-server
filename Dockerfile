FROM python:3.9-slim

WORKDIR /app

# Install system dependencies required by Coqui TTS and its transitive deps
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    build-essential \
    python3-dev \
    libffi-dev \
    libsndfile1 \
    libsndfile1-dev \
    espeak-ng \
    libespeak-ng1 \
    libespeak-ng-dev \
    libopenblas-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip tooling and preinstall CPU Torch (compatible)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu torch==2.1.2

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY xtts_server.py .
COPY dist/reference_kelly.wav dist/
COPY dist/reference_ken_mono16k.wav dist/reference_ken_mono16k.wav
RUN ln -s /app/dist/reference_ken_mono16k.wav /app/dist/reference_ken.wav

# Pre-download the model to speed up cold starts
RUN python -c "from TTS.api import TTS; TTS('tts_models/multilingual/multi-dataset/xtts_v2')"

# Environment
ENV PYTHONUNBUFFERED=1
ENV PORT=5002

EXPOSE 5002

# Health check (simple)
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=5 \
  CMD python - <<'PY' || exit 1\nimport json,urllib.request\ntry:\n  r=urllib.request.urlopen('http://localhost:5002/health',timeout=5)\n  print(r.status)\n  exit(0 if r.status==200 else 1)\nexcept Exception:\n  exit(1)\nPY

CMD ["python", "xtts_server.py"]
