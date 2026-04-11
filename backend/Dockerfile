FROM python:3.12-slim

# Install system dependencies including OpenGL libraries
RUN apt-get update && apt-get install -y \
    libgles2-mesa \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    libusb-1.0-0 \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install uv for faster Python dependency management
RUN pip install uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install Python dependencies
RUN uv sync

# Copy application code
COPY . .

# Download MediaPipe model
RUN wget -q "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task" \
    -O "hand_landmarker.task"

# Expose port
EXPOSE 10000

# Start the application
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000", "--workers", "1"]