import cv2
import argparse
from pathlib import Path
import glob
import os
import json
import numpy as np
import re
import json
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


from parameters import extract_vehicle_size, extract_vehicle_color, detect_plate_text, find_matches
from search_ui import launch_ui 

def save_metadata_to_json(vehicle_events, output_dir="output_data"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    now_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"run_{now_str}.json"
    file_path = Path(output_dir) / filename
    
    output_data = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "total_vehicles": len(vehicle_events)
        },
        "vehicles": {}
    }
    
    for vid, data in vehicle_events.items():
        first = data['first_seen_seconds']
        last = data['last_seen_seconds']
        duration = last - first
        
        output_data["vehicles"][vid] = {
            "vehicle_type": data['vehicle_type'],
            "color": data['color'],
            "first_seen": data['first_seen'],
            "last_seen": data['last_seen'],
            "duration": float(f"{duration:.2f}")
        }
        
    with open(file_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Metadata saved to: {file_path}")


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
                
                # Size Analysis
                size_data = extract_vehicle_size(box, frame.shape)
                
                # Color Analysis
                if track_id in track_colors:
                    color_data = track_colors[track_id]
                else:
                    color_data = extract_vehicle_color(frame, box)
                    if color_data.get("dominant_color") != "unknown":
                        track_colors[track_id] = color_data
                
                # Plate Analysis
                plate_data = detect_plate_text(frame, box)
                
                detections.append({
                    'class_name': VEHICLE_CLASSES[class_id],
                    'confidence': conf,
                    'bbox': (x1, y1, x2, y2),
                    'track_id': track_id,
                    'color_data': color_data,
                    'size_data': size_data,
                    'plate_data': plate_data
                })
    
    return detections

def draw_detections(frame, detections: list):
    annotated_frame = frame.copy()
    
    for det in detections:
        class_name = det['class_name']
        track_id = det.get('track_id', 'N/A')
        
        # Extract new parameters
        color_info = det.get('color_data', {})
        size_info = det.get('size_data', {})
        plate_info = det.get('plate_data', {})
        
        color_name = color_info.get('dominant_color', 'Unknown')
        rgb_color = color_info.get('rgb_value', (255, 255, 255))
        hex_val = color_info.get('hex_value', '')
        size_label = size_info.get('size_class', 'N/A')
        
        plate_text = plate_info.get('plate_text', '')
        plate_found = plate_info.get('plate_detected', False)
        
        x1, y1, x2, y2 = det['bbox']
        
        # Color Handling (RGB to BGR for OpenCV)
        if isinstance(rgb_color, (list, tuple)) and len(rgb_color) == 3:
            bbox_color = (rgb_color[2], rgb_color[1], rgb_color[0])
        else:
            bbox_color = (0, 255, 0)
        
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), bbox_color, 2)
        
        # Label Construction with Hex
        label = f"#{track_id} {color_name} {hex_val}"
        if plate_found and plate_text:
             label += f" [{plate_text}]"
        
        (label_width, label_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        
        # Draw Label Background
        cv2.rectangle(
            annotated_frame,
            (x1, y1 - label_height - 10),
            (x1 + label_width, y1),
            bbox_color,
            -1
        )
        
        # Determine Text Color (Contrast)
        # Luminance: 0.299*R + 0.587*G + 0.114*B
        # bbox_color is BGR
        luminance = 0.299*bbox_color[2] + 0.587*bbox_color[1] + 0.114*bbox_color[0]
        text_color = (0, 0, 0) if luminance > 127 else (255, 255, 255)
        
        # Draw Label Text
        cv2.putText(
            annotated_frame,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            text_color,
            1
        )
    
    return annotated_frame





def process_video(
    input_path: str,
    output_path: str = None,
    model_name: str = "yolov8n.pt",
    confidence_threshold: float = 0.5,
    start_time: str = "00:00:00",
    display: bool = True,
    save: bool = False,
    vehicle_events: dict = None,
    video_start_seconds: float = None,
    video_id_prefix: str = ""
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
    if vehicle_events is None:
        vehicle_events = {}

    if video_start_seconds is None:
        try:
            start_dt = datetime.strptime(start_time, "%H:%M:%S")
            video_start_seconds = 0
        except ValueError:
            print(f"Invalid start time format: {start_time}. Using 00:00:00")
            start_dt = datetime.strptime("00:00:00", "%H:%M:%S")
            video_start_seconds = 0
            
    
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
                    if video_start_seconds is not None and video_start_seconds > 0:
                        current_abs_time = video_start_seconds + elapsed_seconds
                        current_dt = datetime.fromtimestamp(current_abs_time)
                        current_time_str = current_dt.strftime("%Y-%m-%d %H:%M:%S")
                        float_timestamp = current_abs_time
                    else:
                        current_dt = start_dt + timedelta(seconds=elapsed_seconds)
                        current_time_str = current_dt.strftime("%H:%M:%S")
                        float_timestamp = elapsed_seconds 
                    
                    unique_tid = f"{video_id_prefix}{tid}"
                    
                    if unique_tid not in vehicle_events:
                        # Extract string color for event log
                        color_data_obj = det.get('color_data', {})
                        color_str = color_data_obj.get('dominant_color', 'Unknown')
                        rgb_val = color_data_obj.get('rgb_value', None)
                        hex_val = color_data_obj.get('hex_value', '#000000')
                        color_hist = color_data_obj.get('color_histogram', None)
                        
                        plate_data_obj = det.get('plate_data', {})
                        plate_str = plate_data_obj.get('plate_text', '')
                        
                        size_data_obj = det.get('size_data', {})
                        size_class = size_data_obj.get('size_class', 'unknown')
                        aspect_ratio = size_data_obj.get('aspect_ratio', 0.0)
                        
                        vehicle_events[unique_tid] = {
                            'vehicle_id': unique_tid,
                            'vehicle_type': det['class_name'],
                            'color': color_str,
                            'hex_value': hex_val,
                            'rgb_value': rgb_val,
                            'color_histogram': color_hist,
                            'plate_text': plate_str,
                            'size_class': size_class,
                            'aspect_ratio': aspect_ratio,
                            'first_seen': current_time_str,
                            'last_seen': current_time_str,
                            'first_seen_seconds': elapsed_seconds, # relative for specific video duration
                            'last_seen_seconds': elapsed_seconds,
                            'source_video': str(input_path)
                        }
                    else:
                        vehicle_events[unique_tid]['last_seen'] = current_time_str
                        vehicle_events[unique_tid]['last_seen_seconds'] = elapsed_seconds
                        
                        # Update color information continuously (Dynamic Update)
                        color_data_obj = det.get('color_data', {})
                        vehicle_events[unique_tid]['color'] = color_data_obj.get('dominant_color', 'Unknown')
                        vehicle_events[unique_tid]['hex_value'] = color_data_obj.get('hex_value', '#000000')
                        vehicle_events[unique_tid]['rgb_value'] = color_data_obj.get('rgb_value', None)
                        vehicle_events[unique_tid]['color_histogram'] = color_data_obj.get('color_histogram', None)
                        
                        # Update plate text if we found a better one (longer)
                        plate_data_obj = det.get('plate_data', {})
                        new_plate = plate_data_obj.get('plate_text', '')
                        if len(new_plate) > len(vehicle_events[unique_tid].get('plate_text', '')):
                            vehicle_events[unique_tid]['plate_text'] = new_plate
            
            annotated_frame = draw_detections(frame, confirmed_detections)
            
            elapsed_seconds = frame_count / fps
            if video_start_seconds is not None and video_start_seconds > 0:
                 current_abs_time = video_start_seconds + elapsed_seconds
                 current_dt = datetime.fromtimestamp(current_abs_time)
                 current_time_str = current_dt.strftime("%H:%M:%S")
            else:
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
    
    print("Processing Complete!")
    print(f"Total frames processed: {frame_count}")
    print(f"Total unique vehicles tracked: {len(known_vehicles)}")
    
    if video_start_seconds == 0 or video_start_seconds is None:
        print("\n" + "="*20 + " VEHICLE EVENT TIMELINE " + "="*20)
        print(f"{'ID':<10} | {'Type':<10} | {'Color':<10} | {'Duration':<10} | {'First Seen':<20} | {'Last Seen':<20}")
        print("-" * 90)
        
        for vid, data in vehicle_events.items():
            duration = data['last_seen_seconds'] - data['first_seen_seconds']
            print(f"{vid:<10} | {data['vehicle_type']:<10} | {data['color']:<10} | {duration:>8.2f}s | {data['first_seen']:>20} | {data['last_seen']:>20}")
        
        print("="*66)

    if save and output_path:
        print(f"Output saved to: {output_path}")
    print("="*50)


def process_video_sequence(folder_path: str, model_name: str, confidence_threshold: float, display: bool, save: bool):
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
        
    mp4_files = sorted(list(folder.glob("*.mp4")))
    if not mp4_files:
        print(f"No .mp4 files found in {folder}")
        return

    print(f"Found {len(mp4_files)} videos in sequence.")
    
    global_vehicle_events = {}
    
    for i, file_path in enumerate(mp4_files):
        print(f"\n--- Processing Video {i+1}/{len(mp4_files)}: {file_path.name} ---")
        
        match = re.search(r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})", file_path.name)
        
        try:
            if match:
                dt_str = match.group(1)
                dt = datetime.strptime(dt_str, "%Y-%m-%d_%H-%M-%S")
                start_seconds = dt.timestamp()
                print(f"Parsed Start Time: {dt}")
            else:
                print(f"Warning: Could not find timestamp pattern in {file_path.name}. Using system time.")
                start_seconds = datetime.now().timestamp()
        except ValueError:
            print(f"Warning: Could not parse timestamp from {file_path.name}. Using system time.")
            start_seconds = datetime.now().timestamp()
            
        process_video(
            input_path=str(file_path),
            model_name=model_name,
            confidence_threshold=confidence_threshold,
            display=display,
            save=save,
            vehicle_events=global_vehicle_events,
            video_start_seconds=start_seconds,
            video_id_prefix=f"{i}_"
        )
        
    print("\n" + "="*20 + " GLOBAL EVENT TIMELINE " + "="*20)
    print(f"{'ID':<10} | {'Type':<10} | {'Color':<10} | {'Duration':<10} | {'First Seen':<20} | {'Last Seen':<20}")
    print("-" * 90)
    
    sorted_events = sorted(global_vehicle_events.items(), key=lambda item: item[1]['first_seen'])
    
    for vid, data in sorted_events:
        duration = data['last_seen_seconds'] - data['first_seen_seconds']
        print(f"{vid:<10} | {data['vehicle_type']:<10} | {data['color']:<10} | {duration:>8.2f}s | {data['first_seen']:>20} | {data['last_seen']:>20}")
    
    print("="*90)
    
    # Matching Analysis
    print("\n" + "="*20 + " GLOBAL MATCHING REPORT " + "="*20)
    clusters = find_matches(global_vehicle_events)
    
    for i, cluster in enumerate(clusters):
        if len(cluster) > 1:
            print(f"Vehicle Group {i+1}: Found in multiple videos/sequences")
            for vid in cluster:
                data = global_vehicle_events[vid]
                print(f"  - {vid:<10} ({data['color']} {data['vehicle_type']}) at {data['first_seen']}")
            print("-" * 40)
            
    print("="*90)
    
    # Save to JSON
    db_path = "vehicle_db.json"
    with open(db_path, "w") as f:
        json.dump(global_vehicle_events, f, indent=4)
    print(f"Vehicle Database saved to {db_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Vehicle Detection from CCTV Footage using YOLO"
    )
    parser.add_argument(
        "input",
        nargs='?',
        default=None,
        type=str,
        help="Path to input video file or directory"
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
        "--ui",
        action="store_true",
        help="Launch the Vehicle Search UI"
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
        "--save",
        action="store_true",
        help="Enable saving output video (default: False)"
    )
    
    args = parser.parse_args()
    if args.ui:
        print("Launching User Interface...")
        launch_ui()
        return

    if args.input is None:
        parser.print_help()
        return

    # Check if input is directory
    input_path = Path(args.input)
    
    if input_path.is_dir():
        process_video_sequence(
            folder_path=str(input_path),
            model_name=args.model,
            confidence_threshold=args.confidence,
            display=not args.no_display,
            save=args.save
        )
    else:
        if args.start_time is None:
            filename = input_path.name
            args.start_time = VIDEO_START_TIMES.get(filename, "00:00:00")
            if args.start_time != "00:00:00":
                print(f"Auto-detected start time for {filename}: {args.start_time}")
        
        events = {}
        
        process_video(
            input_path=args.input,
            output_path=args.output,
            model_name=args.model,
            confidence_threshold=args.confidence,
            start_time=args.start_time,
            display=not args.no_display,
            save=args.save,
            vehicle_events=events,
        )
        
        save_metadata_to_json(events)


if __name__ == "__main__":
    main()
