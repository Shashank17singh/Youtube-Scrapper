FROM python:3.11-slim

WORKDIR /app

# Copy the entire application
COPY . ./

# Install dependencies using standard pip to avoid any memory/lockfile issues on the free tier builder
RUN pip install --no-cache-dir .

# Provide a default PORT
ENV PORT=7860
EXPOSE $PORT

# Start the application
CMD ["sh", "-c", "ytrag serve --host 0.0.0.0 --port ${PORT}"]
