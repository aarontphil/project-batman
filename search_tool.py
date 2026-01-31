import argparse
import json
import os
import cv2
from pathlib import Path
from difflib import SequenceMatcher

def string_similarity(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def extract_snippet(video_path, start_time, end_time, output_path):
    """
    Extracts a clip from the video between start_time and end_time (seconds).
    """
    if not os.path.exists(video_path):
        print(f"Error: Source video not found: {video_path}")
        return False

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open {video_path}")
        return False
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    current_frame = start_frame
    while current_frame <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)
        current_frame += 1
        
    cap.release()
    writer.release()
    cap.release()
    writer.release()
    return True

def play_video(video_path):
    """
    Plays the video file using OpenCV.
    """
    if not os.path.exists(video_path):
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not play {video_path}")
        return

    print(f"Playing {video_path} (Press 'q' to skip)...")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 25 # Fallback
    delay = int(1000 / fps)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        cv2.imshow("Search Result", frame)
        
        # Press 'q' to skip/exit current video
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="Search for vehicles and extract snippets.")
    parser.add_argument("--db", type=str, default="vehicle_db.json", help="Path to vehicle database JSON")
    parser.add_argument("--color", type=str, help="Filter by color (e.g., red, black)")
    parser.add_argument("--type", type=str, help="Filter by type (car, motorcycle, bus, truck)")
    parser.add_argument("--plate", type=str, help="Filter by partial plate text")
    parser.add_argument("--output", type=str, default="search_results", help="Output directory")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.db):
        print(f"Database not found: {args.db}")
        return
        
    with open(args.db, "r") as f:
        data = json.load(f)
        
    results = []
    
    print(f"Searching {len(data)} records...")
    
    for vid, event in data.items():
        match = True
        
        # Color Filter
        if args.color:
            color_sim = string_similarity(args.color, event.get('color', ''))
            if color_sim < 0.6: # Simple fuzzy threshold
                 match = False
        
        # Type Filter
        if args.type:
            if args.type.lower() not in event.get('vehicle_type', '').lower():
                match = False
                
        # Plate Filter
        if args.plate:
            plate = event.get('plate_text', '')
            if args.plate.upper() not in plate:
                match = False
        
        if match:
            results.append(event)
            
    print(f"Found {len(results)} matches.")
    
    if results:
        os.makedirs(args.output, exist_ok=True)
        
        print("\nExtracting Snippets...")
        for res in results:
            vid_id = res['vehicle_id']
            source = res.get('source_video')
            # Use relative seconds for extraction if available, otherwise absolute logic might be needed
            # Our DB stores 'first_seen_seconds' and 'last_seen_seconds' which are relative to the video file 
            # (as calculated in vehicle_detector.py: process_video)
            start = res.get('first_seen_seconds', 0)
            end = res.get('last_seen_seconds', 0)
            
            # Add padding
            padding = 1.0 # 1 second padding
            start = max(0, start - padding)
            end = end + padding
            
            duration = end - start
            
            print(f"  - ID: {vid_id} | {res['color']} {res['vehicle_type']} | {duration:.1f}s")
            
            clean_id = vid_id.replace(":", "_") # Clean string for filename
            out_file = os.path.join(args.output, f"{clean_id}.mp4")
            
            if source:
                success = extract_snippet(source, start, end, out_file)
                if success:
                    print(f"    Saved to: {out_file}")
                    play_video(out_file)
            else:
                print("    Error: No source video path in record.")

if __name__ == "__main__":
    main()

# Visual Search Extensions
from parameters.vehicle_size import extract_vehicle_size
from parameters.vehicle_color import extract_vehicle_color, extract_color_smart
from parameters.matching import calculate_similarity
from ultralytics import YOLO

MODEL_PATH = "yolov8n.pt"
_model_instance = None

def get_model():
    global _model_instance
    if _model_instance is None:
        _model_instance = YOLO(MODEL_PATH)
    return _model_instance

def analyze_image(image_path):
    """
    Analyzes an uploaded image to extract vehicle features.
    Returns a 'Query Event' dict compatible with vehicle_db.json.
    """
    import logging
    logging.getLogger("ultralytics").setLevel(logging.ERROR)

    if not os.path.exists(image_path):
        return None
        
    img = cv2.imread(image_path)
    if img is None:
        return None
        
    model = get_model()
    # Run inference to find the main vehicle in the image
    results = model(img, verbose=False)[0]
    
    best_box = None
    max_area = 0
    best_cls = None
    
    # Find largest vehicle in the image
    vehicle_classes = [2, 3, 5, 7] # car, motorcycle, bus, truck
    
    if results.boxes:
        for box, cls in zip(results.boxes.xyxy.cpu().numpy(), results.boxes.cls.cpu().numpy()):
            if int(cls) in vehicle_classes:
                x1, y1, x2, y2 = map(int, box)
                area = (x2 - x1) * (y2 - y1)
                if area > max_area:
                    max_area = area
                    best_box = [x1, y1, x2, y2]
                    best_cls = int(cls)
                    
    # If no vehicle found, generate features for the whole image (fallback)
    if best_box is None:
        h, w = img.shape[:2]
        best_box = [0, 0, w, h]
        best_cls = 2 # Assume car if unknown
        
    # Extract Features
    # 1. Color (Use Smart Logic for Uploaded Images)
    color_data = extract_color_smart(img, best_box)
    
    # 2. Size
    size_data = extract_vehicle_size(best_box, img.shape)
    
    # 3. Type
    class_names = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
    v_type = class_names.get(best_cls, "car")
    
    query_event = {
        "vehicle_type": v_type,
        "color": color_data['dominant_color'],
        "rgb_value": color_data['rgb_value'],
        "color_histogram": color_data['color_histogram'],
        "size_class": size_data['size_class'],
        "aspect_ratio": size_data['aspect_ratio'],
        "plate_text": "" 
    }
    
    return query_event, best_box

def search_by_visuals(query_event, db_data, top_k=20):
    """
    Scans the database and finds matches based on visual similarity.
    """
    scored_candidates = []
    
    for vid, event in db_data.items():
        score = calculate_similarity(query_event, event)
        if score > 0.2: 
            scored_candidates.append((score, event))
            
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    
    return scored_candidates[:top_k]
