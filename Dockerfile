FROM python:3.11-slim

WORKDIR /app

# Copy the entire source tree so api/main.py is at /app/api/main.py
# (ytrag serve expects it there relative to the project root)
COPY . ./

# Install only what the serve command needs at runtime.
# faster-whisper (~2GB) and yt-dlp are for local ingestion only — skip them.
RUN pip install --no-cache-dir \
    "qdrant-client>=1.19.0" \
    "groq>=1.6.0" \
    "python-dotenv>=1.2.3" \
    "typer>=0.21.0" \
    "rich>=14.4.0" \
    "fastapi>=0.141.1" \
    "uvicorn>=0.52.1" \
    "pydantic>=2.13.4" \
    "google-genai>=2.19.0" \
    "hatchling" \
 && pip install --no-cache-dir --no-deps -e .

# Provide a default PORT
ENV PORT=7860
EXPOSE $PORT

# ytrag serve finds api/main.py relative to __file__ two levels up = /app/api/main.py
CMD ["sh", "-c", "ytrag serve --host 0.0.0.0 --port ${PORT}"]
