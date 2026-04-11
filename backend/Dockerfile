FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    libgles2-mesa \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install uv

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

RUN wget -q \
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task" \
    -O hand_landmarker.task

COPY . .

EXPOSE 10000

CMD ["uv", "run", "uvicorn", "main:app", \
     "--host", "0.0.0.0", \
     "--port", "10000", \
     "--workers", "1"]