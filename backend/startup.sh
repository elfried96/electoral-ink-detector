#!/bin/bash
set -e

# Installer les dépendances système pour MediaPipe
echo "Installing system dependencies for MediaPipe..."
apt-get update && apt-get install -y \
  libgl1-mesa-glx \
  libglib2.0-0 \
  libsm6 \
  libxext6 \
  libxrender-dev \
  libgomp1 \
  libgles2 \
  mesa-utils-extra \
  || echo "Some packages failed to install, continuing..."

# Télécharger le modèle
MODEL="hand_landmarker.task"
URL="https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
if [ ! -f "$MODEL" ]; then
  echo "Downloading $MODEL..."
  wget -q "$URL" -O "$MODEL" || curl -sL "$URL" -o "$MODEL"
  echo "Model downloaded."
fi

# Configurer l'environnement pour utiliser le rendu software
export MESA_GL_VERSION_OVERRIDE=3.3
export MESA_GLSL_VERSION_OVERRIDE=330
export LIBGL_ALWAYS_SOFTWARE=1

echo "Starting server on port ${PORT:-8000}..."
uv run uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}