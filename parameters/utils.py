import cv2
import numpy as np
import math

# Define standard vehicle colors in RGB format
# (Red, Green, Blue)
# Standard HSV ranges (Hue 0-179, Sat 0-255, Val 0-255)
# No longer using fixed RGB references


def euclidean_distance(c1, c2):
    """
    Calculate Euclidean distance between two colors.
    """
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(c1, c2)))

def get_closest_color_label(rgb_color):
    """
    Finds the color name using HSV ranges.
    rgb_color: (R, G, B) tuple or list
    Returns: (color_name, confidence)
    """
    r, g, b = rgb_color
    
    # Convert RGB to BGR for OpenCV
    # Wrap in numpy array shape (1, 1, 3)
    pixel = np.array([[[b, g, r]]], dtype=np.uint8)
    hsv = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0][0]
    
    h = hsv[0] # 0-179
    s = hsv[1] # 0-255
    v = hsv[2] # 0-255
    
    # 1. Grayscale (Low Saturation)
    # Silver/Gray logic: S is low, V varies
    if s < 40: 
        if v < 40:
            return "black", 0.9
        elif v > 200:
            return "white", 0.9
        elif v > 128:
            return "silver", 0.8 # Lighter gray
        else:
            return "gray", 0.8
            
    # 2. Very Dark (Low Value) - regardless of saturation, looks black/dark
    if v < 40:
        return "black", 0.8
        
    # 3. Hue Mapping
    # Red wraps around 0/180
    if (h >= 0 and h <= 10) or (h >= 170 and h <= 180):
        confidence = 0.9
        return "red", confidence
        
    elif h > 10 and h <= 25:
        return "orange", 0.8
        
    elif h > 25 and h <= 35:
        return "yellow", 0.9
        
    elif h > 35 and h <= 85:
        return "green", 0.9
        
    elif h > 85 and h <= 130:
        return "blue", 0.9
        
    elif h > 130 and h <= 150:
        return "purple", 0.8
        
    elif h > 150 and h < 170:
        return "pink", 0.8
        
    return "unknown", 0.5

def crop_bbox(frame, bbox):
    """
    Crops the frame based on the bounding box.
    bbox expected format: [x1, y1, x2, y2]
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)
    
    # Clip to frame boundaries
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    
    if x1 >= x2 or y1 >= y2:
        return None
        
    return frame[y1:y2, x1:x2]
