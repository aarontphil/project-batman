import cv2
import numpy as np
import re
from parameters.utils import crop_bbox

# Lazy-loaded reader to avoid global import overhead if not used
# or to persist model across calls
_EASYOCR_READER = None

def get_reader():
    global _EASYOCR_READER
    if _EASYOCR_READER is None:
        try:
            import easyocr
            # Initialize for English. 
            # 'gpu=False' to be safe, or True if available. 
            # We'll set gpu=False to maximize compatibility/robustness as requested.
            _EASYOCR_READER = easyocr.Reader(['en'], gpu=False, verbose=False)
        except ImportError:
            return None
    return _EASYOCR_READER

def clean_plate_text(text):
    """
    Cleans OCR text to remove non-alphanumeric characters.
    """
    return re.sub(r'[^A-Z0-9]', '', text.upper())

def detect_plate_text(frame, bbox) -> dict:
    """
    Detects license plate text within a vehicle bounding box.
    Uses a heuristic to focus on the bottom area of the vehicle
    and EasyOCR for text extraction.

    Args:
        frame: Full video frame.
        bbox: Vehicle bounding box [x1, y1, x2, y2].

    Returns:
        Dictionary with detection results.
    """
    
    # 1. Crop Vehicle
    vehicle_roi = crop_bbox(frame, bbox)
    if vehicle_roi is None or vehicle_roi.size == 0:
        return {"plate_detected": False, "plate_text": "", "confidence": 0.0}

    # 2. Heuristic: Plate is usually in the bottom 40% of the vehicle
    h, w = vehicle_roi.shape[:2]
    # If the vehicle is too small, OCR won't work anyway
    if h < 20 or w < 20:
         return {"plate_detected": False, "plate_text": "", "confidence": 0.0}

    search_y = int(h * 0.6) # Start from 60% down
    plate_roi = vehicle_roi[search_y:h, 0:w]
    
    if plate_roi.size == 0:
        plate_roi = vehicle_roi # Fallback

    # 3. Text Detection
    reader = get_reader()
    if reader is None:
        return {
            "plate_detected": False, 
            "plate_text": "", 
            "confidence": 0.0,
            "error": "EasyOCR not installed"
        }

    try:
        # Run OCR
        # detail=0 returns just text, but we want confidence.
        results = reader.readtext(plate_roi)
        
        best_text = ""
        best_conf = 0.0
        details = []

        for res in results:
            # res format: (bbox, text, prob)
            _, text, prob = res
            
            cleaned = clean_plate_text(text)
            
            # Simple filter: Plate should have at least 2 alphanumeric chars
            if len(cleaned) < 2:
                continue
                
            if prob > best_conf:
                best_conf = prob
                best_text = cleaned
            
            details.append({"text": cleaned, "conf": float(prob)})

        detected = len(best_text) > 0

        return {
            "plate_detected": detected,
            "plate_text": best_text,
            "confidence": round(float(best_conf), 2),
            "candidates": details # Optional: helping debug
        }

    except Exception as e:
        return {
            "plate_detected": False, 
            "plate_text": "", 
            "confidence": 0.0,
            "error": str(e)
        }
