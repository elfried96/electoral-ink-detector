FROM python:3.12-slim

# Installer les dépendances système nécessaires pour MediaPipe
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Installer uv
RUN pip install uv

WORKDIR /app

# Copier les fichiers de dépendances
COPY pyproject.toml .
COPY .python-version .

# Installer les dépendances Python avec uv
RUN uv sync

# Copier le reste du code
COPY . .

# Télécharger le modèle MediaPipe
RUN wget -q https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task \
    || curl -sL https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task -o hand_landmarker.task

# Variables d'environnement pour le rendu software OpenGL
ENV MESA_GL_VERSION_OVERRIDE=3.3
ENV MESA_GLSL_VERSION_OVERRIDE=330
ENV LIBGL_ALWAYS_SOFTWARE=1

# Exposer le port
EXPOSE 10000

# Commande de démarrage
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]