"""
Version CPU-only du pipeline qui n'utilise pas OpenGL
Utilise une méthode de détection simplifiée
"""
import cv2
import numpy as np
from typing import Dict, Optional
import mediapipe as mp


def detect_hand_simple(image_rgb: np.ndarray) -> Optional[Dict]:
    """
    Détection simplifiée de main basée sur la couleur de peau
    """
    h, w = image_rgb.shape[:2]
    
    # Convertir en HSV pour détecter la couleur de peau
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    
    # Plages multiples pour différents tons de peau
    # Tons clairs
    lower_skin1 = np.array([0, 10, 60], dtype=np.uint8)
    upper_skin1 = np.array([25, 150, 255], dtype=np.uint8)
    
    # Tons moyens/foncés
    lower_skin2 = np.array([0, 30, 30], dtype=np.uint8)
    upper_skin2 = np.array([35, 255, 255], dtype=np.uint8)
    
    # Créer les masques
    skin_mask1 = cv2.inRange(hsv, lower_skin1, upper_skin1)
    skin_mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
    skin_mask = cv2.bitwise_or(skin_mask1, skin_mask2)
    
    # Nettoyer le masque
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
    
    # Trouver les contours
    contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Prendre le plus grand contour (probablement la main)
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    
    # Vérifier que le contour est assez grand
    if area < (w * h * 0.05):  # Au moins 5% de l'image (plus flexible)
        return None
    
    # Obtenir la boîte englobante
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    return {
        'bbox': (x, y, w, h),
        'mask': skin_mask,
        'area': area
    }


def analyze_ink_simple(image_rgb: np.ndarray) -> Dict:
    """
    Analyse simplifiée pour détecter l'encre violette/bleue
    """
    # Convertir en HSV
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    
    # Détecter les couleurs violettes/bleues (encre) - Plages plus strictes
    # Bleu-violet (encre électorale typique)
    lower_ink1 = np.array([85, 60, 30])  # Bleu avec saturation minimale
    upper_ink1 = np.array([130, 255, 200])
    
    # Violet profond
    lower_ink2 = np.array([130, 60, 30])  # Violet avec saturation minimale
    upper_ink2 = np.array([170, 255, 200])
    
    # Créer les masques
    mask1 = cv2.inRange(hsv, lower_ink1, upper_ink1)
    mask2 = cv2.inRange(hsv, lower_ink2, upper_ink2)
    ink_mask = cv2.bitwise_or(mask1, mask2)
    
    # Nettoyer le masque
    kernel = np.ones((3, 3), np.uint8)
    ink_mask = cv2.morphologyEx(ink_mask, cv2.MORPH_OPEN, kernel)
    ink_mask = cv2.morphologyEx(ink_mask, cv2.MORPH_CLOSE, kernel)
    
    # Calculer le score
    total_pixels = ink_mask.size
    ink_pixels = np.sum(ink_mask > 0)
    ink_ratio = ink_pixels / total_pixels
    
    # Déterminer si encre détectée (seuil plus strict)
    ink_detected = ink_ratio > 0.001  # 0.1% minimum pour le CPU simple
    
    return {
        'ink_detected': ink_detected,
        'score': round(ink_ratio * 100, 2),
        'confidence': min(ink_ratio / 0.001, 1.0) if ink_detected else 0,
        'mask': ink_mask
    }


def run_pipeline_cpu(image_rgb: np.ndarray, sensitivity: float = 1.8) -> Dict:
    """
    Pipeline CPU-only sans MediaPipe ni OpenGL
    """
    try:
        # 1. Détecter la main
        hand_info = detect_hand_simple(image_rgb)
        
        if not hand_info:
            return {
                'success': False,
                'error': 'NO_HAND_DETECTED',
                'message': 'Main non détectée. Montrez votre paume face à la caméra.',
                'ink_detected': False,
                'voted': False,
                'verdict': 'ERREUR',
                'score_global': 0.0,
                'n_doigts_detectes': 0,
                'fraud': {'suspected': False, 'score': 0, 'indicators': []},
                'doigts': {}
            }
        
        # 2. Extraire la région de la main
        x, y, w, h = hand_info['bbox']
        hand_region = image_rgb[y:y+h, x:x+w]
        
        # 3. Analyser l'encre dans la région de la main
        ink_result = analyze_ink_simple(hand_region)
        
        # 4. Déterminer le verdict
        if ink_result['ink_detected']:
            if ink_result['score'] > 5:
                verdict = "CERTAIN"
            elif ink_result['score'] > 2:
                verdict = "PROBABLE"
            else:
                verdict = "POSSIBLE"
            voted = True
        else:
            verdict = "ABSENT"
            voted = False
        
        return {
            'success': True,
            'ink_detected': bool(ink_result['ink_detected']),
            'voted': bool(voted),
            'verdict': verdict,
            'score_global': float(ink_result['score']),
            'n_doigts_detectes': 1 if ink_result['ink_detected'] else 0,
            'fraud': {'suspected': False, 'score': 0, 'indicators': []},
            'doigts': {
                'main': {
                    'ink_detected': bool(ink_result['ink_detected']),
                    'score_pct': float(ink_result['score']),
                    'confidence': float(ink_result['confidence'])
                }
            },
            'method': 'cpu_simple'
        }
        
    except Exception as e:
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