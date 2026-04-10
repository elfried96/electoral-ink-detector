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

from pipeline import run_pipeline, download_model_if_needed, check_image_quality

# Configuration du rate limiting
limiter = Limiter(key_func=get_remote_address)


model_loaded = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_loaded
    try:
        print("Loading hand detection model...")
        download_model_if_needed()
        model_loaded = True
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        model_loaded = False
    yield
    print("Shutting down...")


app = FastAPI(
    title="Ink Detection API",
    description="API pour la détection d'encre électorale sur les doigts",
    version="1.0.0",
    lifespan=lifespan
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


def resize_image_if_needed(image: np.ndarray, max_dimension: int = 1200) -> np.ndarray:
    h, w = image.shape[:2]
    if max(h, w) > max_dimension:
        scale = max_dimension / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return image


@app.get("/")
async def root():
    return {"message": "Ink Detection API v1"}


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "model_loaded": model_loaded
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
        image = Image.open(io.BytesIO(contents))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image_np = np.array(image)
        
        image_np = resize_image_if_needed(image_np, max_dimension=1200)
        
        # Vérifier la qualité de l'image avant l'analyse
        quality_error = check_image_quality(image_np)
        if quality_error:
            return JSONResponse(
                status_code=422,
                content={
                    'success': False,
                    'error':   quality_error['error'],
                    'message': quality_error['message'],
                    'processing_time_ms': int((time.time() - start_time) * 1000)
                }
            )
        
        result = run_pipeline(image_np, sensitivity=sensitivity)
        
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
            "n_doigts_detectes": result.get('n_doigts_detectes', 0),
            "fraud": result.get('fraud', {
                "suspected": False,
                "score": 0,
                "indicators": ["Aucun indicateur suspect"]
            }),
            "doigts": result.get('doigts', {}),
            "processing_time_ms": processing_time_ms
        }
        
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
