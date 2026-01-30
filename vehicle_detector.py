import cv2
import argparse
from pathlib import Path
import numpy as np
from datetime import datetime, timedelta
from ultralytics import YOLO


VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle", 
    5: "bus",
    7: "truck"
}

COLORS = {
    "car": (0, 255, 0),
    "motorcycle": (255, 0, 0),
    "bus": (0, 165, 255),
    "truck": (0, 255, 255)
}

VIDEO_START_TIMES = {
    "videosp.mp4": "07:30:06",
    "sample_traffic.mp4": "13:45:00",
}


def load_model(model_name: str = "yolov8n.pt") -> YOLO:
    print(f"Loading YOLO model: {model_name}")
    model = YOLO(model_name)
    print("Model loaded successfully!")
    return model


def detect_hex_color(crop_img) -> str:
    if crop_img.size == 0:
        return "#000000"

    resized = cv2.resize(crop_img, (50, 50))
    pixels = np.float32(resized.reshape(-1, 3))

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 2
    _, labels, centers = cv2.kmeans(pixels, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)

    centers_hsv = cv2.cvtColor(np.array([centers]), cv2.COLOR_BGR2HSV)[0]
    
    sorted_indices = np.argsort(centers_hsv[:, 1])[::-1]
    dominant_bgr = centers[sorted_indices[0]]

    b, g, r = dominant_bgr
    hex_color = "#{:02x}{:02x}{:02x}".format(r, g, b)
            
    return hex_color


def detect_vehicles(model: YOLO, frame, track_colors: dict, confidence_threshold: float = 0.5) -> list:
    results = model.track(frame, persist=True, verbose=False, tracker="custom_tracker.yaml")[0]
    
    detections = []
    
    if results.boxes.id is not None:
        ids = results.boxes.id.cpu().numpy().astype(int)
        boxes = results.boxes.xyxy.cpu().numpy().astype(int)
        classes = results.boxes.cls.cpu().numpy().astype(int)
        confidences = results.boxes.conf.cpu().numpy().astype(float)
        
        for box, class_id, track_id, conf in zip(boxes, classes, ids, confidences):
            if class_id in VEHICLE_CLASSES and conf >= confidence_threshold:
                x1, y1, x2, y2 = box
                
                if track_id in track_colors:
                    color = track_colors[track_id]
                else:
                    h, w = frame.shape[:2]
                    x1_c, y1_c = max(0, x1), max(0, y1)
                    x2_c, y2_c = min(w, x2), min(h, y2)
                    
                    vehicle_crop = frame[y1_c:y2_c, x1_c:x2_c]
                    color = detect_hex_color(vehicle_crop)
                    track_colors[track_id] = color
                
                
                detections.append({
                    'class_name': VEHICLE_CLASSES[class_id],
                    'confidence': conf,
                    'bbox': (x1, y1, x2, y2),
                    'track_id': track_id,
                    'color': color
                })
    
    return detections


def draw_detections(frame, detections: list):
    annotated_frame = frame.copy()
    
    for det in detections:
        class_name = det['class_name']
        confidence = det['confidence']
        track_id = det.get('track_id', 'N/A')
        vehicle_color = det.get('color', 'Unknown')
        x1, y1, x2, y2 = det['bbox']
        
        hex_str = vehicle_color.lstrip('#')
        if len(hex_str) == 6:
            r, g, b = tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))
            bbox_color = (b, g, r) 
        else:
            bbox_color = (255, 255, 255)
        
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), bbox_color, 2)
        
        label = f"#{track_id} [{vehicle_color}] {class_name}"
        
        (label_width, label_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        
        cv2.rectangle(
            annotated_frame,
            (x1, y1 - label_height - 10),
            (x1 + label_width, y1),
            bbox_color,
            -1
        )
        
        cv2.putText(
            annotated_frame,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2
        )
    
    return annotated_frame


def process_video(
    input_path: str,
    output_path: str = None,
    model_name: str = "yolov8n.pt",
    confidence_threshold: float = 0.5,
    start_time: str = "00:00:00",
    display: bool = True,
    save: bool = True
):
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Video file not found: {input_path}")
    
    model = load_model(model_name)
    
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {input_path}")
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {input_path.name}")
    print(f"Resolution: {width}x{height}, FPS: {fps}, Total frames: {total_frames}")
    
    noise_threshold = max(3, int(fps * 0.5))
    print(f"Adaptive Noise Filter: Removing tracks shorter than {noise_threshold} frames ({noise_threshold/fps:.2f}s)")
    
    writer = None
    if save:
        if output_path is None:
            output_path = input_path.parent / f"{input_path.stem}_detected{input_path.suffix}"
        output_path = Path(output_path)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        print(f"Output will be saved to: {output_path}")
    frame_count = 0
    known_vehicles = set()
    track_colors = {}
    track_history = {}
    vehicle_events = {}
    
    try:
        start_dt = datetime.strptime(start_time, "%H:%M:%S")
    except ValueError:
        print(f"Invalid start time format: {start_time}. Using 00:00:00")
        start_dt = datetime.strptime("00:00:00", "%H:%M:%S")

    print("\nProcessing video... Press 'q' to quit early.")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            detections = detect_vehicles(model, frame, track_colors, confidence_threshold)
            
            confirmed_detections = []
            
            current_ids = []
            for det in detections:
                tid = det['track_id']
                current_ids.append(tid)
                
                track_history[tid] = track_history.get(tid, 0) + 1
                
                if track_history[tid] > noise_threshold:
                    confirmed_detections.append(det)
                    if tid not in known_vehicles:
                        known_vehicles.add(tid)
                    
                        known_vehicles.add(tid)
                    
                    elapsed_seconds = frame_count / fps
                    current_dt = start_dt + timedelta(seconds=elapsed_seconds)
                    current_time_str = current_dt.strftime("%H:%M:%S")
                    
                    if tid not in vehicle_events:
                        vehicle_events[tid] = {
                            'vehicle_id': tid,
                            'vehicle_type': det['class_name'],
                            'color': det['color'],
                            'first_seen': current_time_str,
                            'last_seen': current_time_str,
                            'first_seen_seconds': elapsed_seconds,
                            'last_seen_seconds': elapsed_seconds
                        }
                    else:
                        vehicle_events[tid]['last_seen'] = current_time_str
                        vehicle_events[tid]['last_seen_seconds'] = elapsed_seconds
            
            annotated_frame = draw_detections(frame, confirmed_detections)
            
            elapsed_seconds = frame_count / fps
            current_dt = start_dt + timedelta(seconds=elapsed_seconds)
            current_time_str = current_dt.strftime("%H:%M:%S")


            info_text = f"Time: {current_time_str} | Vehicles: {len(known_vehicles)}"
            cv2.putText(
                annotated_frame,
                info_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )
            
            if writer:
                writer.write(annotated_frame)
            
            if display:
                cv2.imshow("Vehicle Detection", annotated_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nProcessing interrupted by user.")
                    break
            
            if frame_count % 100 == 0:
                print(f"Processed {frame_count}/{total_frames} frames...")
    
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
    
    print("\n" + "="*50)
    print("Processing Complete!")
    print(f"Total frames processed: {frame_count}")
    print(f"Total unique vehicles tracked: {len(known_vehicles)}")
    
    print("\n" + "="*20 + " VEHICLE EVENT TIMELINE " + "="*20)
    print(f"{'ID':<5} | {'Type':<10} | {'Color':<10} | {'Duration':<10} | {'First Seen':<10} | {'Last Seen':<10}")
    print("-" * 75)
    
    for vid, data in vehicle_events.items():
        duration = data['last_seen_seconds'] - data['first_seen_seconds']
        print(f"{vid:<5} | {data['vehicle_type']:<10} | {data['color']:<10} | {duration:>8.2f}s | {data['first_seen']:>10} | {data['last_seen']:>10}")
    
    print("="*66)

    if save and output_path:
        print(f"Output saved to: {output_path}")
    print("="*50)


def main():
    parser = argparse.ArgumentParser(
        description="Vehicle Detection from CCTV Footage using YOLO"
    )
    parser.add_argument(
        "input",
        type=str,
        help="Path to input video file"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Path for output video (default: input_detected.mp4)"
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default="yolov8n.pt",
        choices=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"],
        help="YOLO model variant (default: yolov8n.pt - fastest)"
    )
    parser.add_argument(
        "-c", "--confidence",
        type=float,
        default=0.5,
        help="Confidence threshold for detections (default: 0.5)"
    )
    parser.add_argument(
        "--start-time",
        type=str,
        default=None,
        help="Start time of the video in HH:MM:SS format (default: auto-detected or 00:00:00)"
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable real-time display"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Disable saving output video"
    )
    
    args = parser.parse_args()
    
    if args.start_time is None:
        filename = Path(args.input).name
        args.start_time = VIDEO_START_TIMES.get(filename, "00:00:00")
        if args.start_time != "00:00:00":
            print(f"Auto-detected start time for {filename}: {args.start_time}")
    
    process_video(
        input_path=args.input,
        output_path=args.output,
        model_name=args.model,
        confidence_threshold=args.confidence,
        start_time=args.start_time,
        display=not args.no_display,
        save=not args.no_save
    )


if __name__ == "__main__":
    main()
