#!/bin/bash
set -e

# Forcer MediaPipe en mode CPU pur — pas de GPU requis
export LIBGL_ALWAYS_SOFTWARE=1
export MESA_GL_VERSION_OVERRIDE=3.3
export MEDIAPIPE_DISABLE_GPU=1
export EGL_PLATFORM=surfaceless

MODEL="hand_landmarker.task"
URL="https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
if [ ! -f "$MODEL" ]; then
  echo "Downloading $MODEL..."
  wget -q "$URL" -O "$MODEL" || curl -sL "$URL" -o "$MODEL"
fi
echo "Starting server..."
uv run uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000}