FROM python:3.9-slim

WORKDIR /app

# System dependencies for Coqui TTS and friends
RUN apt-get update && apt-get install -y \
    ffmpeg \
    curl \
    git \
    build-essential \
    python3-dev \
    gfortran \
    libffi-dev \
    libsndfile1 \
    libsndfile1-dev \
    espeak-ng \
    libespeak-ng1 \
    libespeak-ng-dev \
    libopenblas-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and preinstall CPU torch + numpy toolchain to avoid wheel build failures
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu torch==2.1.2 && \
    pip install --no-cache-dir numpy==1.26.4 "cython<3" blis==0.7.11

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App files
COPY xtts_server.py .
COPY dist/reference_kelly.wav dist/
COPY dist/reference_ken_mono16k.wav dist/reference_ken_mono16k.wav
RUN ln -s /app/dist/reference_ken_mono16k.wav /app/dist/reference_ken.wav

# Warm model cache to reduce cold-start
RUN python - <<'PY'
from TTS.api import TTS
TTS('tts_models/multilingual/multi-dataset/xtts_v2')
PY

ENV PYTHONUNBUFFERED=1
EXPOSE 5002

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=5 CMD curl -fsS http://localhost:5002/health || exit 1

CMD ["python", "xtts_server.py"]
