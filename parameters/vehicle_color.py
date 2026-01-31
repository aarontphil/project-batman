import cv2
import numpy as np
from parameters.utils import get_closest_color_label, crop_bbox

def get_color_histogram(image, bins=(8, 8)):
    """
    Calculates 2D HSV histogram (Hue, Saturation).
    Returns normalized flattened histogram.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Calculate histogram for Hue (0-180) and Saturation (0-256)
    # Using 8 bins for each is standard for coarse matching
    hist = cv2.calcHist([hsv], [0, 1], None, bins, [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist.flatten().tolist()

def extract_vehicle_color(frame, bbox) -> dict:
    """
    Extracts the dominant color of the vehicle.

    Args:
        frame: The full video frame (numpy array).
        bbox: [x1, y1, x2, y2] coordinates.

    Returns:
        Dictionary with dominant color name and confidence.
    """
    
    # Crop the vehicle from the frame
    vehicle_roi = crop_bbox(frame, bbox)
    if vehicle_roi is None or vehicle_roi.size == 0:
        return {"dominant_color": "unknown", "confidence": 0.0}

    # Further crop to the center 50% to reduce background noise/road
    h, w = vehicle_roi.shape[:2]
    cx, cy = w // 2, h // 2
    cw, ch = w // 2, h // 2 # Width/Height of crop (50% of original)
    
    start_x = cx - cw // 2
    start_y = cy - ch // 2
    center_roi = vehicle_roi[start_y:start_y+ch, start_x:start_x+cw]
    
    if center_roi.size == 0:
        center_roi = vehicle_roi # Fallback to full ROI if center crop fails

    # Resize for faster processing (K-means can be slow on large images)
    # 64x64 is sufficient for color dominance
    small_roi = cv2.resize(center_roi, (64, 64), interpolation=cv2.INTER_AREA)

    # Reshape to a list of pixels
    pixels = small_roi.reshape(-1, 3)
    pixels = np.float32(pixels)

    # Define criteria for K-means: (type, max_iter, epsilon)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    
    # Run K-means with K=3 to separate vehicle color from windshield/tires/asphalt
    try:
        K = 3
        _, labels, centers = cv2.kmeans(pixels, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Sort centers by "vibrancy" (Saturation * Value) to pick the actual color
        # skipping dull grays/blacks if a colorful option exists.
        best_center = None
        max_score = -1
        
        centers_rgb = []
        
        for center in centers:
            b, g, r = center
            # Convert to HSV to check saturation/value
            # OpenCV expects uint8 0-255 ranges for correct conversion
            c_uint8 = np.array([[[b, g, r]]], dtype=np.uint8)
            hsv = cv2.cvtColor(c_uint8, cv2.COLOR_BGR2HSV)[0][0]
            
            sat = hsv[1]
            val = hsv[2]
            
            # Score: Favor high saturation and decent brightness
            # But don't ignore white/black entirely if they are the ONLY option
            score = int(sat) + int(val) 
            
            # Penalize very dark colors (tires) or very unsaturated colors (road) 
            # unless everything is dark/grey
            if sat < 30: score -= 50
            if val < 30: score -= 50
            
            if score > max_score:
                max_score = score
                best_center = center
                
            centers_rgb.append((center[2], center[1], center[0]))
            
        dominant_color_bgr = best_center if best_center is not None else centers[0]
        
        # --- BLACK/WHITE DISTINCTION FIX ---
        # K-Means often picks up glare (high value) on black cars, making them look white/silver.
        # We check the MEDIAN statistics of the crop to determine the true nature.
        
        # Convert entire ROI to HSV
        full_hsv = cv2.cvtColor(small_roi, cv2.COLOR_BGR2HSV)
        median_sat = np.median(full_hsv[:, :, 1])
        median_val = np.median(full_hsv[:, :, 2])
        
        override_color = None
        
        # If the car is generally low saturation (Grayscale)
        if median_sat < 40:
             # Use Median Value to decide the shade
             if median_val < 60:
                 override_color = "black"
                 dominant_color_bgr = (0, 0, 0) # Force Black
             elif median_val > 180:
                 override_color = "white"
                 dominant_color_bgr = (255, 255, 255) # Force White
             elif median_val > 100:
                 override_color = "silver"
                 dominant_color_bgr = (192, 192, 192)
             else:
                 override_color = "gray"
                 dominant_color_bgr = (128, 128, 128)
                 
        # Convert BGR (OpenCV default) to RGB for our utility
        dominant_color_rgb = (dominant_color_bgr[2], dominant_color_bgr[1], dominant_color_bgr[0]) 
        
        # Helper for Hex
        def rgb_to_hex(rgb):
            return "#{:02x}{:02x}{:02x}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))
            
        hex_val = rgb_to_hex(dominant_color_rgb)
        
        if override_color:
            color_name = override_color
            confidence = 1.0 # High confidence on median-based override
        else:
            color_name, confidence = get_closest_color_label(dominant_color_rgb)
        
        # Calculate Histogram
        hist = get_color_histogram(small_roi)
        
        return {
            "dominant_color": color_name,
            "hex_value": hex_val,
            "confidence": confidence,
            "rgb_value": [int(c) for c in dominant_color_rgb],
            "color_histogram": hist
        }
        
    except Exception as e:
        # Fallback in case of cv2 errors
        return {"dominant_color": "error", "confidence": 0.0, "error": str(e)}

def extract_color_smart(frame, bbox) -> dict:
    """
    Smart color extraction for STATIC UPLOADED IMAGES.
    - Uses strict cropping (removes road/glare).
    - Uses strict pixel filtering (removes background/shadows).
    - Falls back to `extract_vehicle_color` if filtering is too aggressive.
    """
    # 1. Base Crop
    vehicle_roi = crop_bbox(frame, bbox)
    if vehicle_roi is None or vehicle_roi.size == 0:
        return extract_vehicle_color(frame, bbox)

    h, w = vehicle_roi.shape[:2]
    
    # 2. Region-Aware Cropping
    # Remove top 10% (glare), bottom 20% (wheels/road), sides 5% (bg)
    y1 = int(h * 0.10)
    y2 = int(h * 0.80)
    x1 = int(w * 0.05)
    x2 = int(w * 0.95)
    
    smart_roi = vehicle_roi[y1:y2, x1:x2]
    
    if smart_roi.size == 0:
        return extract_vehicle_color(frame, bbox)

    # 3. Pixel Filtering
    # Convert to HSV
    hsv_roi = cv2.cvtColor(smart_roi, cv2.COLOR_BGR2HSV)
    
    # Reshape for masking
    pixels_hsv = hsv_roi.reshape(-1, 3)
    pixels_bgr = smart_roi.reshape(-1, 3)
    
    # Create Mask
    # Condition 1: Saturation < 50 (Road/Shadows/Grey-noise)
    mask_sat = pixels_hsv[:, 1] < 50
    
    # Condition 2: Value < 40 (Dark shadows/Tires)
    mask_val = pixels_hsv[:, 2] < 40
    
    # Condition 3: Sat < 30 AND Val > 200 (White Glare/Sky)
    # Note: This risks removing white cars, but we have fallback.
    mask_white = (pixels_hsv[:, 1] < 30) & (pixels_hsv[:, 2] > 200)
    
    # Combine masks: We want to KEEP pixels that do NOT match these
    noise_mask = mask_sat | mask_val | mask_white
    valid_mask = ~noise_mask
    
    valid_pixels = pixels_bgr[valid_mask]
    
    # 4. Fallback Check
    # If we filtered out almost everything (e.g., < 100 pixels or < 5% of area),
    # it might be a white/black car that we aggressively filtered.
    min_pixels = 100
    if len(valid_pixels) < min_pixels:
        print(f"[SmartColor] Filter too aggressive ({len(valid_pixels)} px). Falling back.")
        return extract_vehicle_color(frame, bbox)
        
    # 5. Clustering on Valid Pixels
    # Resize isn't needed if we work on pixel list directly, 
    # but valid_pixels might be large, so let's cap it? 
    # Actually, K-Means on a few thousand pixels is fast.
    
    pixels = np.float32(valid_pixels)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 3
    
    try:
        # Check if we have enough points for K
        if len(pixels) < K:
             return extract_vehicle_color(frame, bbox)
             
        _, labels, centers = cv2.kmeans(pixels, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Count labels to find dominant cluster
        unique, counts = np.unique(labels, return_counts=True)
        dominant_idx = unique[np.argmax(counts)]
        dominant_center = centers[dominant_idx]
        
        # Convert to int BGR
        b, g, r = int(dominant_center[0]), int(dominant_center[1]), int(dominant_center[2])
        dominant_rgb = (r, g, b)
        
        # Get Label
        color_name, confidence = get_closest_color_label(dominant_rgb)
        
        # Hex
        hex_val = "#{:02x}{:02x}{:02x}".format(r, g, b)
        
        # Dummy Hist (since we filtered pixels, the spatial hist is hard to define exactly, 
        # let's just use the smart_roi hist for compatibility)
        hist = get_color_histogram(smart_roi)
        
        return {
            "dominant_color": color_name,
            "hex_value": hex_val,
            "confidence": confidence,
            "rgb_value": [r, g, b],
            "color_histogram": hist
        }
        
    except Exception as e:
        print(f"[SmartColor] Error: {e}")
        return extract_vehicle_color(frame, bbox)
