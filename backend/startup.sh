#!/bin/bash
set -e

MODEL="hand_landmarker.task"
URL="https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

if [ ! -f "$MODEL" ]; then
  echo "Downloading $MODEL..."
  wget -q "$URL" -O "$MODEL" || curl -sL "$URL" -o "$MODEL"
  echo "Downloaded."
fi

echo "Starting server..."
uv run uvicorn main:app \
  --host 0.0.0.0 \
  --port ${PORT:-10000} \
  --workers 1