FROM python:3.12

# Installer toutes les dépendances système nécessaires pour MediaPipe et OpenGL
RUN apt-get update && apt-get install -y \
    # OpenGL et Mesa
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libglu1-mesa \
    libgles2 \
    libegl1-mesa \
    mesa-utils \
    # X11 et rendu
    libx11-6 \
    libxext6 \
    libxrender1 \
    libxrandr2 \
    libxi6 \
    libxinerama1 \
    libxcursor1 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxtst6 \
    xvfb \
    # Autres dépendances
    libglib2.0-0 \
    libsm6 \
    libgomp1 \
    libgdk-pixbuf2.0-0 \
    libgtk-3-0 \
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

# Variables d'environnement pour forcer le rendu software OpenGL
ENV LIBGL_ALWAYS_SOFTWARE=1
ENV MESA_GL_VERSION_OVERRIDE=3.3
ENV MESA_GLSL_VERSION_OVERRIDE=330
ENV QT_X11_NO_MITSHM=1
ENV DISPLAY=:99

# Créer un script de démarrage avec Xvfb
RUN echo '#!/bin/bash\n\
Xvfb :99 -screen 0 1024x768x24 -nolisten tcp -nolisten unix &\n\
sleep 2\n\
uv run uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000}' > /app/start.sh && \
chmod +x /app/start.sh

# Le port est défini par Render via la variable PORT
EXPOSE ${PORT}

# Utiliser le script de démarrage
CMD ["/app/start.sh"]