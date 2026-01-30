import sys
import os

# Allow running this script directly from parameters/ directory
if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import numpy as np
from ultralytics import YOLO
from parameters.vehicle_size import extract_vehicle_size
from parameters.vehicle_color import extract_vehicle_color
from parameters.plate_text import detect_plate_text
import os

def run_demo(video_path="sample_traffic.mp4"):
    print(f"Starting demo on {video_path}...")
    
    if not os.path.exists(video_path):
        print(f"Error: {video_path} not found.")
        return

    # Load Model
    model = YOLO("yolov8n.pt")
    
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Setup Output
    output_path = "output_analysis.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    vehicle_classes = [2, 3, 5, 7] # car, motorcycle, bus, truck

    frame_count = 0
    max_frames = 300 # Limit for demo purposes
    
    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        if frame_count % 10 == 0:
            print(f"Processing frame {frame_count}...")
            
        # Detect
        results = model.track(frame, persist=True, verbose=False)[0]
        
        if results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy().astype(int)
            ids = results.boxes.id.cpu().numpy().astype(int)
            clss = results.boxes.cls.cpu().numpy().astype(int)
            
            for box, track_id, cls in zip(boxes, ids, clss):
                if cls in vehicle_classes:
                    # 1. Size Analysis
                    size_data = extract_vehicle_size(box, frame.shape)
                    
                    # 2. Color Analysis
                    color_data = extract_vehicle_color(frame, box)
                    
                    # 3. Plate Analysis
                    plate_data = detect_plate_text(frame, box)
                    
                    # Draw Visuals
                    x1, y1, x2, y2 = box
                    color_name = color_data['dominant_color']
                    size_label = size_data['size_class']
                    plate_text = plate_data['plate_text'] if plate_data['plate_detected'] else ""
                    
                    # Bounding Box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Labels
                    label = f"ID:{track_id} {size_label} {color_name}"
                    if plate_text:
                        label += f" [{plate_text}]"
                        
                    cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        writer.write(frame)

    cap.release()
    writer.release()
    print(f"Analysis complete. Saved to {output_path}")

if __name__ == "__main__":
    run_demo()
