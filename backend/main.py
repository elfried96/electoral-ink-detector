import io
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from pipeline import run_pipeline, preprocess_image, HAND_LANDMARKER

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
@limiter.limit("10/minute")
async def analyze_image(request: Request, file: UploadFile = File(...)):
    import time
    start_time = time.time()

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_rgb = np.array(image)
    except Exception:
        raise HTTPException(status_code=400, detail={
            "error": "IMAGE_INVALID",
            "message": "Fichier image non lisible."
        })

    result = run_pipeline(image_rgb)

    if not result.get("success"):
        raise HTTPException(status_code=422, detail={
            "error":   result.get("error", "UNKNOWN"),
            "message": result.get("message", "Analyse échouée.")
        })

    result["processing_time_ms"] = int((time.time() - start_time) * 1000)
    return result


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
