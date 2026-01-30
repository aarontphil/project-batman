import cv2
import argparse
from pathlib import Path
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


def load_model(model_name: str = "yolov8n.pt") -> YOLO:
    print(f"Loading YOLO model: {model_name}")
    model = YOLO(model_name)
    print("Model loaded successfully!")
    return model


def detect_vehicles(model: YOLO, frame, confidence_threshold: float = 0.5) -> list:
    results = model(frame, verbose=False)[0]
    
    detections = []
    
    for box in results.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        
        if class_id in VEHICLE_CLASSES and confidence >= confidence_threshold:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            detections.append({
                'class_name': VEHICLE_CLASSES[class_id],
                'confidence': confidence,
                'bbox': (x1, y1, x2, y2)
            })
    
    return detections


def draw_detections(frame, detections: list):
    annotated_frame = frame.copy()
    
    for det in detections:
        class_name = det['class_name']
        confidence = det['confidence']
        x1, y1, x2, y2 = det['bbox']
        
        color = COLORS.get(class_name, (255, 255, 255))
        
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
        
        label = f"{class_name}: {confidence:.2f}"
        
        (label_width, label_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        
        cv2.rectangle(
            annotated_frame,
            (x1, y1 - label_height - 10),
            (x1 + label_width, y1),
            color,
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
    
    writer = None
    if save:
        if output_path is None:
            output_path = input_path.parent / f"{input_path.stem}_detected{input_path.suffix}"
        output_path = Path(output_path)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        print(f"Output will be saved to: {output_path}")
    
    frame_count = 0
    total_detections = 0
    
    print("\nProcessing video... Press 'q' to quit early.")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            detections = detect_vehicles(model, frame, confidence_threshold)
            total_detections += len(detections)
            
            annotated_frame = draw_detections(frame, detections)
            
            info_text = f"Frame: {frame_count}/{total_frames} | Vehicles: {len(detections)}"
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
    print(f"Total vehicle detections: {total_detections}")
    print(f"Average detections per frame: {total_detections/frame_count:.2f}")
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
    
    process_video(
        input_path=args.input,
        output_path=args.output,
        model_name=args.model,
        confidence_threshold=args.confidence,
        display=not args.no_display,
        save=not args.no_save
    )


if __name__ == "__main__":
    main()
