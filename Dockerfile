FROM python:3.11-slim

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies into the system python. 
# We skip installing the "cuda" extra by default to keep the image small,
# since transcription (faster-whisper) usually runs locally or on a separate GPU server,
# and HuggingFace Spaces free tier only provides CPU.
RUN uv sync --frozen --no-dev --system

# Copy the rest of the application
COPY . ./

# Provide a default PORT
ENV PORT=7860
EXPOSE $PORT

# Start the application using the custom CLI
CMD ["sh", "-c", "uv run ytrag serve --host 0.0.0.0 --port ${PORT}"]
