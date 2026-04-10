import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from typing import Dict, List, Tuple, Optional, Any
import os
import urllib.request


def download_model_if_needed(model_path: str = "hand_landmarker.task"):
    if not os.path.exists(model_path):
        print(f"Downloading {model_path}...")
        url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        urllib.request.urlretrieve(url, model_path)
        print(f"{model_path} downloaded successfully!")
    return model_path


def normalize_light(image_rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    
    lab = cv2.merge([l_channel, a_channel, b_channel])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def check_image_quality(image_rgb: np.ndarray) -> Optional[Dict]:
    """
    Vérifie la qualité de l'image AVANT l'analyse.
    Retourne None si OK, ou un dict d'erreur sinon.
    """
    h, w = image_rgb.shape[:2]
    
    # Taille minimum
    if w < 400 or h < 400:
        return {
            'error': 'IMAGE_TOO_SMALL',
            'message': 'Photo trop petite. Approchez-vous de la caméra.'
        }
    
    # Netteté (variance du Laplacian)
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    print(f"[DEBUG] Sharpness: {sharpness}")
    # Seuil TRÈS bas - on accepte presque tout après prétraitement  
    if sharpness < 10:  # Seuil TRÈS bas (était 30, puis 80)
        return {
            'error': 'IMAGE_BLURRY',
            'message': f'Image vraiment trop floue (score: {sharpness:.1f}).'
        }
    
    # Luminosité (canal V)
    hsv = cv2.cvtColor(
        cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR),
        cv2.COLOR_BGR2HSV
    )
    brightness = float(hsv[:,:,2].mean())
    if brightness < 60:
        return {
            'error': 'BAD_LIGHTING',
            'message': 'Image trop sombre. Allez vers une fenêtre ou activez le flash.'
        }
    if brightness > 230:
        return {
            'error': 'BAD_LIGHTING',
            'message': 'Image surexposée. Évitez le flash direct.'
        }
    
    return None  # OK


def get_palm_color(image_rgb: np.ndarray, hand_landmarks: List) -> Dict:
    h, w = image_rgb.shape[:2]
    
    palm_indices = [0, 1, 5, 9, 13, 17]
    palm_points = []
    
    for idx in palm_indices:
        if idx < len(hand_landmarks):
            landmark = hand_landmarks[idx]
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            palm_points.append((x, y))
    
    if len(palm_points) < 3:
        return None
    
    mask = np.zeros((h, w), dtype=np.uint8)
    hull = cv2.convexHull(np.array(palm_points))
    cv2.fillPoly(mask, [hull], 255)
    
    kernel = np.ones((15, 15), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    
    palm_pixels = hsv[mask > 0]
    if len(palm_pixels) == 0:
        return None
    
    h_median = np.median(palm_pixels[:, 0])
    s_median = np.median(palm_pixels[:, 1])
    v_median = np.median(palm_pixels[:, 2])
    
    return {
        'h': h_median,
        's': s_median,
        'v': v_median,
        'mask': mask
    }


def crop_finger(image_rgb: np.ndarray, hand_landmarks: List, 
                finger_name: str, margin: int = 35) -> Tuple[Optional[np.ndarray], Optional[Tuple]]:
    h, w = image_rgb.shape[:2]
    
    finger_ranges = {
        'pouce': [1, 2, 3, 4],
        'index': [5, 6, 7, 8]
    }
    
    if finger_name not in finger_ranges:
        return None, None
    
    indices = finger_ranges[finger_name]
    xs = []
    ys = []
    
    for idx in indices:
        if idx < len(hand_landmarks):
            landmark = hand_landmarks[idx]
            xs.append(int(landmark.x * w))
            ys.append(int(landmark.y * h))
    
    if len(xs) < 2:
        return None, None
    
    x1 = max(0, min(xs) - margin)
    y1 = max(0, min(ys) - margin)
    x2 = min(w, max(xs) + margin)
    y2 = min(h, max(ys) + margin)
    
    if x2 <= x1 or y2 <= y1:
        return None, None
    
    crop = image_rgb[y1:y2, x1:x2].copy()
    
    # Exclure la zone ongle (25% supérieurs du crop)
    h_crop = crop.shape[0]
    crop = crop[int(h_crop * 0.25):, :]
    
    return crop, (x1, y1, x2, y2)


def analyze_ink_adaptive(region_rgb: np.ndarray, palm_color: Dict, 
                         sensitivity: float = 1.8) -> Dict:
    if region_rgb is None or region_rgb.size == 0 or palm_color is None:
        return {
            'ink_detected': False,
            'score': 0.0,
            'confidence': 0.0,
            'method': 'adaptive_hsv_constrained',
            'mask': None,
            'details': {}
        }
    
    bgr = cv2.cvtColor(region_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h_ch = hsv[:,:,0].astype(float)
    s_ch = hsv[:,:,1].astype(float)
    v_ch = hsv[:,:,2].astype(float)
    
    # Contrainte 1 : teinte obligatoirement bleue ou violette
    INK_HUE_MIN, INK_HUE_MAX = 85, 170
    correct_hue = (h_ch >= INK_HUE_MIN) & (h_ch <= INK_HUE_MAX)
    
    # Contrainte 2 : plus sombre que la paume
    v_thresh = max(25, v_ch.std() * sensitivity)
    is_darker = v_ch < (palm_color['v'] - v_thresh)
    
    # Contrainte 3 : plus saturé que la paume
    s_thresh = max(25, s_ch.std() * sensitivity)
    is_richer = s_ch > (palm_color['s'] + s_thresh)
    
    # Contrainte 4 : exclure fond blanc / flash
    not_background = v_ch < 235
    
    # Pixel = encre SEULEMENT si les 3 conditions sont réunies
    ink_mask = correct_hue & (is_darker | is_richer) & not_background
    ink_mask = ink_mask.astype(np.uint8) * 255
    
    kern = np.ones((3,3), np.uint8)
    ink_mask = cv2.morphologyEx(ink_mask, cv2.MORPH_OPEN,  kern)
    ink_mask = cv2.morphologyEx(ink_mask, cv2.MORPH_CLOSE, kern)
    
    score = float(np.sum(ink_mask > 0)) / ink_mask.size
    SEUIL = 0.05  # 5% minimum
    
    return {
        'ink_detected': score >= SEUIL,
        'score':        round(score * 100, 2),  # Convertir en pourcentage
        'confidence':   round(min(score / SEUIL, 1.0), 3),
        'method':       'adaptive_hsv_constrained',
        'mask':         ink_mask,
        'details': {
            'palm_h': round(palm_color['h'], 1),
            'palm_s': round(palm_color['s'], 1),
            'palm_v': round(palm_color['v'], 1),
            'v_thresh': round(v_thresh, 1),
            's_thresh': round(s_thresh, 1),
            'ink_hue_range': f'{INK_HUE_MIN}-{INK_HUE_MAX}'
        }
    }


def score_final(finger_results: Dict[str, Dict]) -> Dict:
    pouce_result = finger_results.get('pouce', {})
    index_result = finger_results.get('index', {})
    
    pouce_detected = pouce_result.get('ink_detected', False)
    index_detected = index_result.get('ink_detected', False)
    
    pouce_score = pouce_result.get('score', 0)
    index_score = index_result.get('score', 0)
    
    # Score moyen entre les deux doigts
    score_global = (pouce_score * 0.5) + (index_score * 0.5)
    
    n_doigts = int(pouce_detected) + int(index_detected)
    
    if n_doigts == 2:
        verdict = "CERTAIN"
        voted = True
    elif n_doigts == 1:
        verdict = "PROBABLE"
        voted = True
    else:
        verdict = "ABSENT"
        voted = False
    
    return {
        'score_global': round(score_global, 2),
        'n_doigts_detectes': n_doigts,
        'verdict': verdict,
        'voted': voted,
        'ink_detected': voted
    }


def detect_fraud(image_rgb: np.ndarray) -> Dict:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    score = 0
    inds = []
    
    # Variance (image synthétique = trop uniforme)
    var = float(np.var(gray))
    if var < 100:
        inds.append('Texture trop uniforme — image peut-être synthétique')
        score += 25
    
    # Laplacian (retouche = bords sur-nets)
    lap_var = float(np.var(cv2.Laplacian(gray, cv2.CV_64F)))
    if lap_var > 10000:
        inds.append('Netteté anormale détectée')
        score += 20
    
    # ELA (Error Level Analysis)
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    _, enc = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 75])
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    ela = cv2.absdiff(bgr, dec)
    ela_score = float(ela.mean())
    if ela_score > 12:
        inds.append('Anomalies de compression détectées')
        score += 30
    
    if not inds:
        inds.append('Aucun indicateur suspect détecté')
    
    # Seuil monté à 65 (était 50 — trop de faux positifs)
    return {
        'suspected': score >= 65,
        'score': min(score, 100),
        'indicators': inds
    }


def run_pipeline(image_rgb: np.ndarray, sensitivity: float = 1.8) -> Dict:
    try:
        image_rgb = normalize_light(image_rgb)
        
        model_path = download_model_if_needed()
        
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.2,  # Réduit de 0.4 à 0.2
            min_hand_presence_confidence=0.2,   # Réduit de 0.4 à 0.2
            min_tracking_confidence=0.2         # Réduit de 0.4 à 0.2
        )
        
        detector = mp_vision.HandLandmarker.create_from_options(options)
        
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        detection_result = detector.detect(mp_image)
        
        if not detection_result.hand_landmarks:
            print(f"[ERROR] Aucune main détectée. Image shape: {image_rgb.shape}")
            # Essayons avec différents prétraitements
            
            # Tentative 1: Image originale sans normalisation
            try:
                print("[RETRY] Tentative sans normalisation de lumière...")
                mp_image_raw = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
                detection_result = detector.detect(mp_image_raw)
                
                if detection_result.hand_landmarks:
                    print("[SUCCESS] Main détectée sans normalisation!")
                    hand_landmarks = detection_result.hand_landmarks[0]
                else:
                    # Tentative 2: Augmenter le contraste
                    print("[RETRY] Tentative avec augmentation du contraste...")
                    enhanced = cv2.convertScaleAbs(image_rgb, alpha=1.5, beta=30)
                    mp_image_enhanced = mp.Image(image_format=mp.ImageFormat.SRGB, data=enhanced)
                    detection_result = detector.detect(mp_image_enhanced)
                    
                    if detection_result.hand_landmarks:
                        print("[SUCCESS] Main détectée avec contraste augmenté!")
                        hand_landmarks = detection_result.hand_landmarks[0]
                        image_rgb = enhanced  # Utiliser l'image améliorée pour la suite
            except Exception as e:
                print(f"[ERROR] Retry failed: {e}")
            
            # Si toujours pas de main détectée
            if not detection_result.hand_landmarks:
                fraud_result = detect_fraud(image_rgb)
                return {
                    'success': False,
                    'error': 'NO_HAND_DETECTED',
                    'message': 'Main non détectée. Montrez votre paume ouverte face à la caméra.',
                    'ink_detected': False,
                    'voted': False,
                    'verdict': 'ERREUR',
                    'score_global': 0.0,
                    'n_doigts_detectes': 0,
                    'fraud': {
                        'suspected': bool(fraud_result.get('suspected', False)),
                        'score': int(fraud_result.get('score', 0)),
                        'indicators': list(fraud_result.get('indicators', []))
                    },
                    'doigts': {}
                }
        else:
            hand_landmarks = detection_result.hand_landmarks[0]
        
        palm_color = get_palm_color(image_rgb, hand_landmarks)
        
        if palm_color is None:
            fraud_result = detect_fraud(image_rgb)
            return {
                'success': False,
                'error': 'Impossible d\'extraire la couleur de la paume',
                'ink_detected': False,
                'voted': False,
                'verdict': 'ERREUR',
                'score_global': 0.0,
                'n_doigts_detectes': 0,
                'fraud': {
                    'suspected': bool(fraud_result.get('suspected', False)),
                    'score': int(fraud_result.get('score', 0)),
                    'indicators': list(fraud_result.get('indicators', []))
                },
                'doigts': {}
            }
        
        finger_results = {}
        
        for finger_name in ['pouce', 'index']:
            finger_region, bbox = crop_finger(image_rgb, hand_landmarks, finger_name)
            if finger_region is not None:
                finger_results[finger_name] = analyze_ink_adaptive(
                    finger_region, palm_color, sensitivity
                )
            else:
                finger_results[finger_name] = {
                    'ink_detected': False,
                    'score': 0.0,
                    'confidence': 0.0,
                    'details': f'{finger_name} non trouvé'
                }
        
        final_score = score_final(finger_results)
        fraud_result = detect_fraud(image_rgb)
        
        return {
            'success': True,
            'ink_detected': bool(final_score['ink_detected']),
            'voted': bool(final_score['voted']),
            'verdict': str(final_score['verdict']),
            'score_global': float(final_score['score_global']),
            'n_doigts_detectes': int(final_score['n_doigts_detectes']),
            'fraud': {
                'suspected': bool(fraud_result.get('suspected', False)),
                'score': int(fraud_result.get('score', 0)),
                'indicators': list(fraud_result.get('indicators', []))
            },
            'doigts': {
                name: {
                    'ink_detected': bool(result['ink_detected']),
                    'score_pct': float(result['score']),
                    'confidence': float(result['confidence'])
                }
                for name, result in finger_results.items()
            },
            'palm_color': {
                'h': float(palm_color['h']),
                's': float(palm_color['s']),
                'v': float(palm_color['v'])
            }
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
            'fraud': {
                'suspected': False,
                'score': 0,
                'indicators': []
            },
            'doigts': {}
        }