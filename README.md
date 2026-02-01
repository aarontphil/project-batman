# Project Cypher: Vehicle Detection & Search System

A comprehensive computer vision system designed to detect, track, and catalog vehicles from CCTV footage. The system utilizes YOLOv8 for accurate specific object detection and provides a graphical user interface (UI) for searching and reviewing vehicle sightings based on attributes like color, type, and license plate.

## Features

- **Advanced Vehicle Detection**: Uses YOLOv8 (You Only Look Once) to partially detect and track vehicles (cars, motorcycles, buses, trucks) in video streams.
- **Attribute Extraction**:
  - **Color Detection**: Identifies dominant vehicle colors.
  - **Size Estimation**: Categorizes vehicles by size.
  - **License Plate Recognition**: Detects and reads license plate text.
- **Search & Playback UI**:
  - Filter results by date, time, vehicle type, color, and license plate.
  - Visual "Routemap" to see detection sequences.
  - Instant video playback of specific detection events.
- **Data Persistence**: Saves detection metadata to JSON for analysis and historical search.
- **Batch Processing**: Capable of processing individual video files or entire directories of footage.

## Installation

1. **Clone the repository** (if applicable) or navigate to the project directory.

2. **Install Dependencies**:
   Ensure you have Python 3.8+ installed. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
   *Key dependencies include: `ultralytics` (YOLOv8), `opencv-python`, and `numpy`.*

## Usage

### 1. Vehicle Detection (CLI)

Run the detection script on a video file to generate metadata and (optionally) a processed video.

```bash
python vehicle_detector.py <input_path> [options]
```

**Examples:**

*   **Basic detection on a single video:**
    ```bash
    python vehicle_detector.py videos/traffic_sample.mp4
    ```

*   **Process a video and save the output with bounding boxes:**
    ```bash
    python vehicle_detector.py videos/traffic_sample.mp4 --save
    ```

*   **Process an entire directory of videos:**
    ```bash
    python vehicle_detector.py videos/ --save
    ```

**Command Line Arguments:**
- `input`: Path to input video file or directory.
- `-o`, `--output`: Path for output video file.
- `-m`, `--model`: YOLO model variant to use (default: `yolov8n.pt`). Options: `yolov8n.pt`, `yolov8s.pt`, `yolov8m.pt`, etc.
- `-c`, `--confidence`: Confidence threshold for detections (default: `0.5`).
- `--start-time`: Manual start time for the video (HH:MM:SS).
- `--no-display`: Run without the real-time video preview window (faster).
- `--save`: Save the processed video with annotations to disk.
- `--ui`: Launch the Search UI directly.

### 2. Search & Review Interface (GUI)

Launch the graphical interface to search through the database of detected vehicles (`vehicle_db.json`).

```bash
python vehicle_detector.py --ui
```
*Alternatively, you can run:*
```bash
python search_ui.py
```

**UI Features:**
- **Search Filters**: Use the dropdowns and text fields to filter by Vehicle Type, Color, or Plate Number.
- **Results List**: Click on a result to see details.
- **Playback Controls**:
  - **Play Clip**: Plays the specific segment where the vehicle was detected.
  - **Play Full Video**: Opens the original source video.
- **Generate Routemap**: Visualizes the sequence of detections.

## Project Structure

- `vehicle_detector.py`: Main script for detection logic and processing pipeline.
- `search_ui.py`: Tkinter-based GUI for searching and viewing results.
- `search_tool.py`: Helper functions for search relevance and string matching.
- `parameters/`: Directory containing analysis modules:
  - `plate_text.py`: OCR logic for license plates.
  - `utils.py`, `vehicle_analysis.py`: Helper utilities for size and color extraction.
- `vehicle_db.json`: The database file storing aggregated vehicle entries.
- `output_data/`: Directory where individual run logs are stored.

## Requirements

- Python 3.x
- `ultralytics`
- `opencv-python`
- `numpy`
