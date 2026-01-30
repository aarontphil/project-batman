
# Configurable thresholds for size classification
# Based on ratio of vehicle area to total frame area
SIZE_THRESHOLDS = {
    "small_limit": 0.05,   # Below 5% of frame area is Small
    "medium_limit": 0.20   # Below 20% (but above 5%) is Medium, else Large
}

def calculate_area(w, h):
    return w * h

def extract_vehicle_size(bbox, frame_shape) -> dict:
    """
    Estimates vehicle size metrics.
    
    Args:
        bbox: List or tuple [x1, y1, x2, y2]
        frame_shape: Tuple (height, width, channels) or (height, width)
        
    Returns:
        Dictionary containing size metrics.
    """
    
    # Unpack inputs
    x1, y1, x2, y2 = bbox
    frame_h, frame_w = frame_shape[:2]
    
    # Calculate dimensions
    width = x2 - x1
    height = y2 - y1
    
    # Basic validation
    if width <= 0 or height <= 0:
        return {
            "bbox_area": 0,
            "aspect_ratio": 0.0,
            "size_ratio": 0.0,
            "size_class": "unknown"
        }

    # Metrics
    area = calculate_area(width, height)
    frame_area = calculate_area(frame_w, frame_h)
    
    aspect_ratio = round(width / height, 2)
    size_ratio = area / frame_area if frame_area > 0 else 0.0
    
    # Classification
    size_class = "large"
    if size_ratio < SIZE_THRESHOLDS["small_limit"]:
        size_class = "small"
    elif size_ratio < SIZE_THRESHOLDS["medium_limit"]:
        size_class = "medium"
        
    return {
        "bbox_area": int(area),
        "aspect_ratio": aspect_ratio,
        "size_ratio": round(size_ratio, 4),
        "size_class": size_class
    }
