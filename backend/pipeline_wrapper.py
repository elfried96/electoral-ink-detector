"""
Wrapper pour choisir entre MediaPipe et CPU pipeline
"""
from typing import Dict
import numpy as np


def run_pipeline(image_rgb: np.ndarray, sensitivity: float = 1.8) -> Dict:
    """
    Essaye d'abord MediaPipe, puis bascule sur CPU si échec
    """
    # Essayer MediaPipe d'abord
    try:
        from pipeline import run_pipeline_mediapipe
        result = run_pipeline_mediapipe(image_rgb, sensitivity)
        if result.get('success') is not False or 'libGL' not in result.get('error', ''):
            return result
    except Exception as e:
        error_msg = str(e)
        print(f"[WARNING] MediaPipe failed: {error_msg}")
        
        # Si c'est une erreur OpenGL, utiliser le fallback CPU
        if 'libGL' in error_msg or 'libEGL' in error_msg or 'cannot open shared object' in error_msg:
            print("[INFO] OpenGL error detected, switching to CPU-only pipeline")
    
    # Fallback sur pipeline avancé sans MediaPipe
    try:
        from pipeline_yolo import run_pipeline_yolo
        print("[INFO] Using advanced CV pipeline (no MediaPipe)")
        return run_pipeline_yolo(image_rgb, sensitivity)
    except ImportError:
        # Si scipy/skimage ne sont pas installés, utiliser le pipeline CPU simple
        from pipeline_cpu import run_pipeline_cpu
        print("[INFO] Using simple CPU pipeline (no MediaPipe)")
        return run_pipeline_cpu(image_rgb, sensitivity)
    except Exception as e:
        print(f"[ERROR] All pipelines failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'ink_detected': False,
            'voted': False,
            'verdict': 'ERREUR',
            'score_global': 0.0,
            'n_doigts_detectes': 0,
            'fraud': {'suspected': False, 'score': 0, 'indicators': []},
            'doigts': {}
        }