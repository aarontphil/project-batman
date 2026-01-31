from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import cv2
import os
import tempfile
from datetime import datetime

def extract_high_res_frame(source_video, timestamp_seconds):
    """Extracts a frame from the video at the given timestamp."""
    if not os.path.exists(source_video):
        return None
        
    cap = cv2.VideoCapture(source_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 30
    
    frame_idx = int(timestamp_seconds * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        # Convert to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame
    return None

def generate_case_report(event_data, output_path="case_file.pdf"):
    """
    Generates a PDF report for the given vehicle event.
    """
    c = canvas.Canvas(output_path, pagesize=letter)
    width, height = letter
    
    # 1. Header
    c.setFont("Helvetica-Bold", 24)
    c.drawCentredString(width / 2, height - 50, "CASE REPORT")
    
    # 2. Suspect Image
    img_array = extract_high_res_frame(
        event_data.get('source_video', ''), 
        event_data.get('first_seen_seconds', 0)
    )
    
    # [Image drawing logic remains the same, assuming y_cursor logic handles the removed header lines implicitly or I explicitly set it]
    # Actually, previous code set y_cursor relative to image.
    # Let's keep the image placement logic but maybe adjust top margin if needed.
    
    if img_array is not None:
        # Save temp image for ReportLab
        fd, temp_img = tempfile.mkstemp(suffix=".jpg")
        os.close(fd)
        
        # Resize for report (keep aspect ratio)
        # Target width: 400
        h, w, _ = img_array.shape
        aspect = h / w
        target_w = 400
        target_h = int(target_w * aspect)
        
        bgr_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        cv2.imwrite(temp_img, bgr_img)
        
        # Draw
        c.drawImage(temp_img, (width - target_w) / 2, height - 120 - target_h, width=target_w, height=target_h)
        
        try:
            os.remove(temp_img)
        except:
            pass
            
        y_cursor = height - 120 - target_h - 40
    else:
        c.drawString(50, height - 200, "[IMAGE NOT AVAILABLE]")
        y_cursor = height - 300
        
    # 3. Details Table
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y_cursor, "DETAILS")
    y_cursor -= 20
    
    c.setFont("Helvetica", 12)
    details = [
        ("ID:", f"{event_data.get('vehicle_id', 'UNK')}"),
        ("Date/Time:", event_data.get('first_seen', 'Unknown')),
        ("Location:", os.path.basename(event_data.get('source_video', 'Unknown'))),
        ("Vehicle Type:", event_data.get('vehicle_type', 'Unknown').upper()),
        ("Color:", event_data.get('color', 'Unknown').upper()),
        ("Duration:", f"{event_data.get('last_seen_seconds', 0) - event_data.get('first_seen_seconds', 0):.2f} seconds")
    ]
    
    for label, value in details:
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y_cursor, label)
        c.setFont("Helvetica", 12)
        c.drawString(150, y_cursor, value)
        y_cursor -= 20
        
    y_cursor -= 20
    c.line(50, y_cursor, width - 50, y_cursor)
    y_cursor -= 30
    
    # 4. Notes
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y_cursor, "NOTES")
    y_cursor -= 20
    c.setFont("Helvetica", 12)
    note = "Vehicle identified via automated visual surveillance system. " \
           "Match confidence derived from visual color histogram and aspect ratio analysis."
           
    text_obj = c.beginText(50, y_cursor)
    text_obj.setFont("Helvetica", 12)
    # Simple wrapping
    words = note.split()
    line = []
    for word in words:
        line.append(word)
        if len(" ".join(line)) > 70:
            text_obj.textLine(" ".join(line))
            line = []
    if line:
        text_obj.textLine(" ".join(line))
        
    c.drawText(text_obj)
    
    # Footer
    c.setFont("Helvetica-Oblique", 10)
    c.drawCentredString(width / 2, 30, f"Generated Report | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    c.save()
    return True
