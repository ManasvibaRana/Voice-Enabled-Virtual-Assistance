# syntax=docker/dockerfile:1
FROM python:3.11-slim

# ffmpeg/libgomp: PyAV (audio decoding) and ctranslate2 (whisper).
# espeak-ng: pyttsx3 has no voice at all on Linux without it, and a voice
# assistant that cannot speak in its own container is not much of a demo.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg libgomp1 espeak-ng \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first, so editing source does not invalidate the pip layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the models into the image. Without this the first question after every
# `docker run` pays a ~150MB download; with it, startup is a few seconds.
ENV HF_HOME=/models
RUN python -c "\
from faster_whisper import WhisperModel; \
WhisperModel('base', device='cpu', compute_type='int8'); \
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('all-MiniLM-L6-v2'); \
print('models cached')"

COPY . .

# Seed the database at build time so the image runs out of the box.
RUN python data/seed_db.py

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
