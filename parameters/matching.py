import numpy as np
import cv2
from difflib import SequenceMatcher
from difflib import SequenceMatcher
from parameters.utils import euclidean_distance

def string_similarity(a, b):
    """Returns a ratio of similarity between two strings (0.0 - 1.0)."""
    return SequenceMatcher(None, a, b).ratio()

def calculate_similarity(event1, event2):
    """
    Calculates the probability that two vehicle events represent the same physical vehicle.
    
    Args:
        event1, event2: Dictionaries containing vehicle data 
                        (type, color_rgb, plate_text, size_class)
        
    Returns:
        float: Similarity score (0.0 to 1.0)
    """
    
    # 1. HARD FILTER: Vehicle Type
    # If types are mostly distinct (e.g. Bus vs Motorcycle), return 0.
    # Exception: Car vs Truck can sometimes be ambiguous in YOLO, but let's be strict for now.
    if event1.get('vehicle_type') != event2.get('vehicle_type'):
        return 0.0

    score = 0.0
    weights = 0.0
    
    # 2. LICENSE PLATE (Strong Indication)
    plate1 = event1.get('plate_text', "")
    plate2 = event2.get('plate_text', "")
    
    if plate1 and plate2:
        # If both identified a plate, compare them
        plate_sim = string_similarity(plate1, plate2)
        if plate_sim > 0.8:
            return 1.0 # High confidence match regardless of color (plates are unique)
        elif plate_sim < 0.4:
             return 0.0 # Distinct plates -> Distinct vehicles
    
    
    # 3. COLOR SIMILARITY (Visual) - RGB Distance
    # Compare RGB values
    rgb1 = event1.get('rgb_value')
    rgb2 = event2.get('rgb_value')
    
    if rgb1 and rgb2:
        dist = euclidean_distance(rgb1, rgb2)
        # Weight 0.3 for dominant color
        color_sim = max(0.0, 1.0 - (dist / 100.0)) 
        score += color_sim * 0.3 
        weights += 0.3
        
    # 4. HISTOGRAM SIMILARITY (Visual) - Distribution
    hist1 = event1.get('color_histogram')
    hist2 = event2.get('color_histogram')
    
    if hist1 and hist2:
        # Correlation: 1.0 is perfect match, 0 is no match, -1 is inverse
        # We clamp negative to 0
        h1 = np.array(hist1, dtype=np.float32)
        h2 = np.array(hist2, dtype=np.float32)
        hist_sim = cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)
        hist_sim = max(0.0, hist_sim)
        
        score += hist_sim * 0.4 # 40% Weight (Very strong indicator)
        weights += 0.4

    # 5. SIZE & SHAPE SIMILARITY (Visual)
    # Compare size classes
    size1 = event1.get('size_class')
    size2 = event2.get('size_class')
    
    if size1 and size2:
        if size1 == size2:
            size_sim = 1.0
        elif (size1 == 'small' and size2 == 'medium') or (size1 == 'medium' and size2 == 'large'):
             size_sim = 0.5 
        else:
             size_sim = 0.0 
             
        score += size_sim * 0.1 # Reduced weight to 10%
        weights += 0.1

    # Aspect Ratio Check
    ar1 = event1.get('aspect_ratio')
    ar2 = event2.get('aspect_ratio')
    
    if ar1 and ar2:
        diff = abs(ar1 - ar2)
        if diff > 0.5: # Significant shape difference
             return 0.2 * score # Heavy Penalty
        elif diff < 0.1:
             score += 1.0 * 0.2 # boost
             weights += 0.2

    if weights == 0:
        return 0.0
        
    final_score = score / weights
    return round(final_score, 3)

def find_matches(all_events):
    """
    Groups events into clusters of unique vehicles.
    
    Args:
        all_events: Dict {event_id: event_data}
        
    Returns:
        list of lists: [[id1, id3], [id2], [id4, id5]]
    """
    event_ids = list(all_events.keys())
    n = len(event_ids)
    visited = set()
    clusters = []
    
    # Simple transitive clustering
    for i in range(n):
        id_i = event_ids[i]
        if id_i in visited:
            continue
            
        current_cluster = [id_i]
        visited.add(id_i)
        
        # Look for matches with remaining items
        # Note: This is a greedy approach. A proper graph connectivity check is better 
        # but O(N^2) is acceptable for typical traffic density.
        
        # We check all UNVISITED items to see if they match ANY in the current cluster
        # But to keep it simple: just checking against the seed (id_i) is usually enough 
        # if the similarity function is robust. 
        # For better chains (A=B, B=C -> A=C), we should check the whole cluster, 
        # but let's stick to seed matching for stability.
        
        for j in range(i + 1, n):
            id_j = event_ids[j]
            if id_j in visited:
                continue
                
            sim = calculate_similarity(all_events[id_i], all_events[id_j])
            
            # Threshold for Visual Match
            if sim > 0.85: 
                current_cluster.append(id_j)
                visited.add(id_j)
                
        clusters.append(current_cluster)
        
    return clusters
