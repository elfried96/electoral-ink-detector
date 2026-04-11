from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import cv2
import numpy as np
from PIL import Image
import io
import time
from typing import Dict, Any
import os

from pipeline import run_pipeline_mediapipe as run_pipeline, preprocess_image, HAND_LANDMARKER

# Configuration du rate limiting
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="Ink Detection API",
    description="API pour la détection d'encre électorale sur les doigts",
    version="1.0.0"
)

# Ajouter le handler pour rate limit personnalisé
app.state.limiter = limiter

@app.exception_handler(RateLimitExceeded)
async def custom_rate_limit_handler(request: Request, exc: RateLimitExceeded):
    response = JSONResponse(
        status_code=429,
        content={
            "error": "TOO_MANY_REQUESTS",
            "message": "Trop de tentatives. Attendez 1 minute."
        }
    )
    response.headers["Retry-After"] = "60"
    return response

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.get("/")
async def root():
    return {"message": "Ink Detection API v1"}


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "model_loaded": HAND_LANDMARKER is not None,
        "delegate": "CPU"
    }


@app.post("/analyze")
@limiter.limit("10/minute;100/hour")
async def analyze_image(
    request: Request,
    file: UploadFile = File(...),
    sensitivity: float = 1.8
) -> Dict[str, Any]:
    if not model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please try again later."
        )
    
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image"
        )
    
    try:
        start_time = time.time()
        
        contents = await file.read()
        print(f"[INFO] Received image, size: {len(contents)} bytes")
        image = Image.open(io.BytesIO(contents))
        print(f"[INFO] Image opened successfully: {image.size}, mode: {image.mode}")
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image_np = np.array(image)
        h, w = image_np.shape[:2]
        
        # Vérification minimale : image trop petite pour être analysée
        if h < 100 or w < 100:
            return JSONResponse(
                status_code=422,
                content={
                    'success': False,
                    'error': 'IMAGE_TOO_SMALL',
                    'message': f'Image vraiment trop petite ({w}x{h}). Minimum requis: 100x100',
                    'processing_time_ms': int((time.time() - start_time) * 1000)
                }
            )
        
        print(f"[INFO] Image shape: {image_np.shape}")
        
        result = run_pipeline(image_np, sensitivity=sensitivity)
        print(f"[INFO] Pipeline result: success={result.get('success')}, error={result.get('error')}")
        
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        if not result.get('success', False):
            # Cas spécifique : aucune main détectée
            if result.get('error') == 'Aucune main détectée':
                return JSONResponse(
                    status_code=422,
                    content={
                        'success': False,
                        'error':   'NO_HAND_DETECTED',
                        'message': 'Main non détectée. Montrez votre paume ouverte, pouce et index bien visibles.',
                        'processing_time_ms': processing_time_ms
                    }
                )
            return JSONResponse(
                status_code=422,
                content={
                    "success": False,
                    "error": result.get('error', 'Unknown error'),
                    "processing_time_ms": processing_time_ms
                }
            )
        
        response = {
            "success": True,
            "ink_detected": result.get('ink_detected', False),
            "voted": result.get('voted', False),
            "verdict": result.get('verdict', 'ABSENT'),
            "score_global": result.get('score_global', 0),
            "doigt_encre": result.get('best_finger', None),  # quel doigt a l'encre
            "fraud": result.get('fraud', {
                "suspected": False,
                "score": 0,
                "indicators": ["Aucun indicateur suspect"]
            }),
            "doigts": result.get('doigts', {}),
            "processing_time_ms": processing_time_ms
        }
        
        # Ajouter le rapport de prétraitement s'il existe
        if 'preprocessing' in result:
            response['preprocessing'] = result['preprocessing']
        
        if 'palm_color' in result:
            response['palm_color'] = result['palm_color']
        
        return response
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": f"Internal server error: {str(e)}",
                "processing_time_ms": int((time.time() - start_time) * 1000) if 'start_time' in locals() else 0
            }
        )


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
