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


def preprocess_image(image_rgb: np.ndarray) -> tuple[np.ndarray, Dict]:
    """
    Prétraitement automatique de l'image AVANT l'analyse.
    Corrige tous les problèmes courants sans rejeter l'image.
    Retourne : (image_corrigee, rapport_corrections)
    """
    corrections = []
    img = image_rgb.copy()
    h, w = img.shape[:2]

    # ── 1. REDIMENSIONNEMENT ──────────────────────────────────
    # Trop petite → agrandir (minimum 640px sur le grand côté)
    # Trop grande → réduire  (maximum 1200px sur le grand côté)
    MIN_SIZE = 640
    MAX_SIZE = 1200
    grand_cote = max(h, w)

    if grand_cote < MIN_SIZE:
        scale = MIN_SIZE / grand_cote
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        corrections.append(f"Agrandie {w}x{h} → {new_w}x{new_h}")

    elif grand_cote > MAX_SIZE:
        scale = MAX_SIZE / grand_cote
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        corrections.append(f"Réduite {w}x{h} → {new_w}x{new_h}")

    # ── 2. CORRECTION LUMINOSITÉ ─────────────────────────────
    # CLAHE sur canal L (LAB) — corrige sous/surexposition
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    brightness = float(l.mean())

    if brightness < 80 or brightness > 210:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        img = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)
        if brightness < 80:
            corrections.append(f"Luminosité corrigée (trop sombre: {brightness:.0f})")
        else:
            corrections.append(f"Luminosité corrigée (surexposée: {brightness:.0f})")
    else:
        # Appliquer CLAHE léger même si luminosité OK
        # pour uniformiser les conditions d'analyse
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        img = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)

    # ── 3. CORRECTION FLOU ───────────────────────────────────
    # Si image floue → appliquer un filtre de netteté (unsharp mask)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    if sharpness < 100:
        # Unsharp mask : accentue les contours sans dégrader
        gaussian = cv2.GaussianBlur(img, (0, 0), 3.0)
        img = cv2.addWeighted(img, 1.5, gaussian, -0.5, 0)
        img = np.clip(img, 0, 255).astype(np.uint8)
        corrections.append(f"Netteté améliorée (score: {sharpness:.0f})")

    # ── 4. CORRECTION ROTATION ───────────────────────────────
    # Détecter si l'image est en portrait ou paysage
    # Si paysage (largeur > hauteur) → rotation 90° pour avoir
    # la main en portrait (plus facile pour MediaPipe)
    h2, w2 = img.shape[:2]
    if w2 > h2 * 1.2:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        corrections.append(f"Rotation 90° (paysage → portrait)")

    # ── 5. RÉDUCTION BRUIT ───────────────────────────────────
    # Filtre bilatéral : réduit le bruit en préservant les bords
    # Important pour les photos prises en mouvement
    img = cv2.bilateralFilter(img, d=5, sigmaColor=35, sigmaSpace=35)

    # ── 6. RAPPORT FINAL ─────────────────────────────────────
    h_final, w_final = img.shape[:2]
    rapport = {
        'corrections_appliquees': corrections,
        'taille_originale': f"{w}x{h}",
        'taille_finale': f"{w_final}x{h_final}",
        'n_corrections': len(corrections)
    }

    return img, rapport


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
    """
    Calcule le score final basé sur les résultats des doigts.
    L'encre électorale est apposée sur UN SEUL doigt (pouce OU index).
    Il suffit qu'UN des deux soit positif pour confirmer le vote.
    """
    pouce_ink = finger_results.get('pouce', {}).get('ink_detected', False)
    index_ink = finger_results.get('index', {}).get('ink_detected', False)

    pouce_score = finger_results.get('pouce', {}).get('score', 0)
    index_score = finger_results.get('index', {}).get('score', 0)

    # Le doigt avec le meilleur score est le doigt marqué
    best_score = max(pouce_score, index_score)
    best_finger = 'index' if index_score >= pouce_score else 'pouce'

    # UN seul doigt positif = vote confirmé
    has_ink = pouce_ink or index_ink

    return {
        'ink_detected': has_ink,
        'verdict': 'CERTAIN' if has_ink else 'ABSENT',
        'score_global': round(best_score, 5),
        'best_finger': best_finger if has_ink else None,  # quel doigt a l'encre
        'n_doigts_detectes': 1 if has_ink else 0,
        'voted': has_ink
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


def run_pipeline_mediapipe(image_rgb: np.ndarray, sensitivity: float = 1.8) -> Dict:
    try:
        # Étape 1 : prétraitement automatique
        image_preprocessed, rapport = preprocess_image(image_rgb)
        print(f"[PREPROCESS] {rapport['n_corrections']} corrections appliquées")
        
        # Étape 2 : détection main sur image corrigée
        try:
            model_path = download_model_if_needed()
            
            base_options = mp_python.BaseOptions(
                model_asset_path=model_path,
                delegate=mp_python.BaseOptions.Delegate.CPU
            )
            options = mp_vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=mp_vision.RunningMode.IMAGE,
                num_hands=2,
                min_hand_detection_confidence=0.4,
                min_hand_presence_confidence=0.4,
                min_tracking_confidence=0.4
            )
            
            detector = mp_vision.HandLandmarker.create_from_options(options)
            
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_preprocessed)
            detection_result = detector.detect(mp_image)
        except Exception as e:
            print(f"[ERROR] MediaPipe initialization failed: {e}")
            raise
        
        if not detection_result.hand_landmarks:
            print(f"[WARNING] Première tentative échouée. Image shape: {image_preprocessed.shape}")
            # Une seule tentative de retry avec plus de contraste
            try:
                print("[RETRY] Augmentation du contraste...")
                enhanced = cv2.convertScaleAbs(image_preprocessed, alpha=1.8, beta=40)
                mp_image_enhanced = mp.Image(image_format=mp.ImageFormat.SRGB, data=enhanced)
                detection_result = detector.detect(mp_image_enhanced)
                
                if detection_result.hand_landmarks:
                    print("[SUCCESS] Main détectée après augmentation du contraste!")
                    hand_landmarks = detection_result.hand_landmarks[0]
                    image_preprocessed = enhanced  # Utiliser l'image améliorée pour la suite
            except Exception as e:
                print(f"[ERROR] Retry failed: {e}")
            
            # Si toujours pas de main détectée
            if not detection_result.hand_landmarks:
                fraud_result = detect_fraud(image_preprocessed)
                return {
                    'success': False,
                    'error': 'NO_HAND_DETECTED',
                    'message': 'Main non détectée. Montrez votre paume ouverte face à la caméra.',
                    'preprocessing': rapport,
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
        
        palm_color = get_palm_color(image_preprocessed, hand_landmarks)
        
        if palm_color is None:
            fraud_result = detect_fraud(image_preprocessed)
            return {
                'success': False,
                'error': 'Impossible d\'extraire la couleur de la paume',
                'preprocessing': rapport,
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
            finger_region, bbox = crop_finger(image_preprocessed, hand_landmarks, finger_name)
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
        fraud_result = detect_fraud(image_preprocessed)
        
        return {
            'success': True,
            'ink_detected': bool(final_score['ink_detected']),
            'voted': bool(final_score['voted']),
            'verdict': str(final_score['verdict']),
            'score_global': float(final_score['score_global']),
            'best_finger': final_score.get('best_finger'),  # quel doigt a l'encre
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
            },
            'preprocessing': rapport  # Ajouter le rapport de prétraitement
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