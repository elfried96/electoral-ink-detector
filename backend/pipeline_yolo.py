"""
Pipeline utilisant une détection de main basée sur contours et couleur avancée
Sans dépendance à MediaPipe ou OpenGL
"""
import cv2
import numpy as np
from typing import Dict, Optional, Tuple
from scipy import ndimage
from skimage import morphology, measure


def enhance_image_for_detection(image_rgb: np.ndarray) -> np.ndarray:
    """Améliore l'image pour une meilleure détection"""
    # Augmenter le contraste et la luminosité
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l_channel, a, b = cv2.split(lab)
    
    # CLAHE sur le canal L
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    
    enhanced = cv2.merge([l_channel, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
    
    return enhanced


def detect_hand_advanced(image_rgb: np.ndarray) -> Optional[Dict]:
    """
    Détection avancée de main en combinant plusieurs méthodes
    """
    h, w = image_rgb.shape[:2]
    
    # 1. Améliorer l'image
    enhanced = enhance_image_for_detection(image_rgb)
    
    # 2. Convertir en plusieurs espaces de couleur
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_RGB2HSV)
    ycrcb = cv2.cvtColor(enhanced, cv2.COLOR_RGB2YCrCb)
    
    # 3. Détection de peau multi-espace
    # HSV
    lower_hsv = np.array([0, 10, 60], dtype=np.uint8)
    upper_hsv = np.array([25, 173, 229], dtype=np.uint8)
    mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)
    
    # YCrCb
    lower_ycrcb = np.array([0, 133, 77], dtype=np.uint8)
    upper_ycrcb = np.array([235, 173, 127], dtype=np.uint8)
    mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)
    
    # Combiner les masques
    skin_mask = cv2.bitwise_and(mask_hsv, mask_ycrcb)
    
    # 4. Nettoyer le masque avec des opérations morphologiques avancées
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    
    # Supprimer le bruit
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel_small)
    # Combler les trous
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel_large)
    
    # Dilater pour connecter les régions proches
    skin_mask = cv2.dilate(skin_mask, kernel_small, iterations=2)
    
    # 5. Trouver les contours
    contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # 6. Analyser les contours pour trouver la main
    best_contour = None
    best_score = 0
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Filtrer par taille
        if area < (w * h * 0.02):  # Au moins 2% de l'image
            continue
        
        # Calculer des caractéristiques du contour
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
            
        # Circularité (une main n'est pas trop circulaire)
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        # Convex hull pour détecter les doigts
        hull = cv2.convexHull(contour, returnPoints=False)
        if len(hull) > 3 and len(contour) > 3:
            defects = cv2.convexityDefects(contour, hull)
            if defects is not None:
                num_defects = len(defects)
                # Une main a généralement 1-4 défauts de convexité (entre les doigts)
                if 0 <= num_defects <= 5:
                    score = area / (w * h) + (1 - abs(circularity - 0.5))
                    if score > best_score:
                        best_score = score
                        best_contour = contour
    
    if best_contour is None:
        # Prendre le plus grand contour comme fallback
        best_contour = max(contours, key=cv2.contourArea)
    
    # 7. Obtenir la région de la main
    x, y, w_box, h_box = cv2.boundingRect(best_contour)
    
    # Agrandir légèrement la boîte
    margin = 20
    x = max(0, x - margin)
    y = max(0, y - margin)
    w_box = min(w - x, w_box + 2 * margin)
    h_box = min(h - y, h_box + 2 * margin)
    
    return {
        'bbox': (x, y, w_box, h_box),
        'mask': skin_mask,
        'contour': best_contour,
        'area': cv2.contourArea(best_contour)
    }


def extract_finger_regions(image_rgb: np.ndarray, hand_info: Dict) -> Dict[str, np.ndarray]:
    """
    Extrait les régions des doigts de la main détectée
    """
    x, y, w, h = hand_info['bbox']
    hand_region = image_rgb[y:y+h, x:x+w]
    
    # Diviser la région en zones pour les doigts
    h_region, w_region = hand_region.shape[:2]
    
    fingers = {}
    
    # Zone du pouce (partie gauche)
    thumb_region = hand_region[:, :w_region//3]
    fingers['pouce'] = thumb_region
    
    # Zone de l'index (partie centrale haute)
    if h_region > 100:
        index_region = hand_region[:h_region//2, w_region//3:2*w_region//3]
        fingers['index'] = index_region
    
    # Zone générale des doigts (partie supérieure)
    fingers_region = hand_region[:h_region//2, :]
    fingers['doigts'] = fingers_region
    
    return fingers


def analyze_ink_advanced(image_rgb: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict:
    """
    Analyse avancée pour détecter l'encre électorale
    """
    h, w = image_rgb.shape[:2]
    
    # Convertir en plusieurs espaces de couleur
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    
    # Détection d'encre multi-critères
    
    # 1. Détection HSV stricte (bleu-violet)
    lower_ink_hsv = np.array([85, 80, 30])
    upper_ink_hsv = np.array([170, 255, 200])
    mask_hsv = cv2.inRange(hsv, lower_ink_hsv, upper_ink_hsv)
    
    # 2. Détection LAB (tons bleus-violets)
    # Dans LAB, b < 0 indique du bleu
    l_channel, a_channel, b_channel = cv2.split(lab)
    mask_lab = (b_channel < 110).astype(np.uint8) * 255  # Valeurs bleues
    
    # 3. Détection par différence avec la couleur de peau moyenne
    if mask is not None:
        # Calculer la couleur moyenne de la peau
        skin_pixels = image_rgb[mask > 0]
        if len(skin_pixels) > 0:
            skin_mean = np.mean(skin_pixels, axis=0)
            # Détecter les pixels très différents de la peau
            diff = np.linalg.norm(image_rgb - skin_mean, axis=2)
            mask_diff = (diff > 50).astype(np.uint8) * 255
        else:
            mask_diff = np.zeros((h, w), dtype=np.uint8)
    else:
        mask_diff = np.zeros((h, w), dtype=np.uint8)
    
    # 4. Combiner les masques
    ink_mask = cv2.bitwise_and(mask_hsv, mask_lab)
    if mask is not None:
        ink_mask = cv2.bitwise_and(ink_mask, mask_diff)
    
    # 5. Nettoyer le masque
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    ink_mask = cv2.morphologyEx(ink_mask, cv2.MORPH_OPEN, kernel)
    ink_mask = cv2.morphologyEx(ink_mask, cv2.MORPH_CLOSE, kernel)
    
    # 6. Analyse des composants connectés
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(ink_mask, connectivity=8)
    
    # Filtrer les petits composants (bruit)
    min_size = 10  # pixels minimum
    filtered_mask = np.zeros_like(ink_mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            filtered_mask[labels == i] = 255
    
    # 7. Calculer les métriques
    total_pixels = h * w
    ink_pixels = np.sum(filtered_mask > 0)
    ink_ratio = ink_pixels / total_pixels if total_pixels > 0 else 0
    
    # 8. Déterminer le verdict
    ink_detected = ink_ratio > 0.001  # 0.1% minimum
    
    # Calculer un score de confiance basé sur plusieurs facteurs
    confidence = 0
    if ink_detected:
        # Facteur de taille
        size_factor = min(ink_ratio / 0.01, 1.0)  # Normaliser à 1% comme maximum
        
        # Facteur de concentration (l'encre devrait être concentrée, pas dispersée)
        if num_labels > 1:
            largest_component = max(stats[1:, cv2.CC_STAT_AREA])
            concentration_factor = largest_component / ink_pixels if ink_pixels > 0 else 0
        else:
            concentration_factor = 1.0
        
        confidence = (size_factor * 0.7 + concentration_factor * 0.3)
    
    return {
        'ink_detected': bool(ink_detected),
        'score': round(ink_ratio * 100, 3),
        'confidence': float(min(confidence, 1.0)),
        'num_regions': max(0, num_labels - 1),
        'mask': filtered_mask
    }


def run_pipeline_yolo(image_rgb: np.ndarray, sensitivity: float = 1.8) -> Dict:
    """
    Pipeline avancé sans MediaPipe ni OpenGL
    """
    try:
        print("[YOLO] Démarrage du pipeline avancé...")
        
        # 1. Détecter la main
        hand_info = detect_hand_advanced(image_rgb)
        
        if not hand_info:
            print("[YOLO] Aucune main détectée")
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
        
        print(f"[YOLO] Main détectée : bbox={hand_info['bbox']}")
        
        # 2. Extraire les régions des doigts
        finger_regions = extract_finger_regions(image_rgb, hand_info)
        
        # 3. Analyser l'encre dans chaque région
        results = {}
        total_score = 0
        max_score = 0
        
        for finger_name, finger_region in finger_regions.items():
            if finger_region.size > 0:
                ink_result = analyze_ink_advanced(finger_region)
                results[finger_name] = ink_result
                total_score += ink_result['score']
                max_score = max(max_score, ink_result['score'])
                
                if ink_result['ink_detected']:
                    print(f"[YOLO] Encre détectée sur {finger_name}: {ink_result['score']}%")
        
        # 4. Déterminer le verdict global
        avg_score = total_score / len(results) if results else 0
        any_ink = any(r['ink_detected'] for r in results.values())
        
        if any_ink:
            if max_score > 5:
                verdict = "CERTAIN"
            elif max_score > 2:
                verdict = "PROBABLE"
            elif max_score > 0.5:
                verdict = "POSSIBLE"
            else:
                verdict = "TRACES"
            voted = True
        else:
            verdict = "ABSENT"
            voted = False
        
        print(f"[YOLO] Verdict: {verdict}, Score max: {max_score}%")
        
        return {
            'success': True,
            'ink_detected': bool(any_ink),
            'voted': bool(voted),
            'verdict': verdict,
            'score_global': float(max_score),
            'n_doigts_detectes': sum(1 for r in results.values() if r['ink_detected']),
            'fraud': {'suspected': False, 'score': 0, 'indicators': []},
            'doigts': {
                name: {
                    'ink_detected': bool(result['ink_detected']),
                    'score_pct': float(result['score']),
                    'confidence': float(result['confidence'])
                }
                for name, result in results.items()
            },
            'method': 'advanced_cv'
        }
        
    except Exception as e:
        print(f"[YOLO] Erreur: {e}")
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