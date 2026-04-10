# API de Détection d'Encre Électorale

API FastAPI pour détecter la présence d'encre électorale sur les doigts (pouce et index) à partir d'images.

## Installation

### Prérequis
- Python 3.12
- uv (gestionnaire de packages)

### Installation de uv
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Configuration du projet
```bash
cd backend
uv python pin 3.12
uv sync
```

## Lancement

### Mode développement
```bash
uv run uvicorn main:app --reload --port 8000
```

### Mode production (Render)
```bash
./startup.sh
```

## Endpoints

### GET /
Retourne le message de bienvenue

### GET /health
Vérifie le statut de l'API et le chargement du modèle

### POST /analyze
Analyse une image pour détecter l'encre électorale

**Paramètres:**
- `file`: Image à analyser (multipart/form-data)
- `sensitivity`: Sensibilité de détection (défaut: 1.8)

**Réponse:**
```json
{
  "success": true,
  "ink_detected": true,
  "voted": true,
  "verdict": "CERTAIN",
  "score_global": 7.4,
  "n_doigts_detectes": 2,
  "fraud": {
    "suspected": false,
    "score": 0,
    "indicators": ["Aucun indicateur suspect"]
  },
  "doigts": {
    "pouce": {
      "ink_detected": true,
      "score_pct": 9.1,
      "confidence": 1.0
    },
    "index": {
      "ink_detected": false,
      "score_pct": 1.2,
      "confidence": 0.4
    }
  },
  "processing_time_ms": 230
}
```

## Architecture

- **main.py**: API FastAPI avec endpoints et gestion du lifecycle
- **pipeline.py**: Logique de détection d'encre avec MediaPipe
- **startup.sh**: Script de démarrage pour Render (télécharge le modèle si nécessaire)

## Technologies

- **FastAPI**: Framework web
- **MediaPipe**: Détection des landmarks de la main
- **OpenCV**: Traitement d'image
- **NumPy**: Calculs numériques
- **Pillow**: Manipulation d'images
- **uv**: Gestion des dépendances