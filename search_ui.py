import tkinter as tk
from tkinter import ttk, messagebox
import json
import os
import cv2
import tempfile
import threading
from datetime import datetime
from search_tool import extract_snippet, string_similarity

class VehicleSearchApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Vehicle Search & Playback")
        self.root.geometry("800x600")
        
        self.db_path = "vehicle_db.json"
        self.data = {}
        self.matches = []
        self.is_playing = False
        
        # Load Data
        self.load_database()
        
        # UI Components
        self.create_widgets()
        
    def load_database(self):
        if os.path.exists(self.db_path):
            with open(self.db_path, "r") as f:
                self.data = json.load(f)
        else:
            messagebox.showerror("Error", f"Database not found: {self.db_path}")

    def create_widgets(self):
        # Input Frame
        input_frame = ttk.LabelFrame(self.root, text="Search Criteria", padding="10")
        input_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(input_frame, text="Color:").grid(row=0, column=0, padx=5)
        self.entry_color = ttk.Entry(input_frame)
        self.entry_color.grid(row=0, column=1, padx=5)
        
        ttk.Label(input_frame, text="Type:").grid(row=0, column=2, padx=5)
        self.entry_type = ttk.Entry(input_frame)
        self.entry_type.grid(row=0, column=3, padx=5)
        
        ttk.Label(input_frame, text="Plate:").grid(row=0, column=4, padx=5)
        self.entry_plate = ttk.Entry(input_frame)
        self.entry_plate.grid(row=0, column=5, padx=5)

        ttk.Label(input_frame, text="Time (HH:MM):").grid(row=0, column=6, padx=5)
        self.entry_time = ttk.Entry(input_frame, width=10)
        self.entry_time.grid(row=0, column=7, padx=5)
        
        ttk.Button(input_frame, text="Search", command=self.perform_search).grid(row=0, column=8, padx=10)
        
        # Results List
        list_frame = ttk.LabelFrame(self.root, text="Search Results", padding="10")
        list_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        columns = ("ID", "Type", "Color", "Time", "Duration", "Source")
        self.tree = ttk.Treeview(list_frame, columns=columns, show="headings")
        
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=100)
            
        self.tree.pack(fill="both", expand=True, side="left")
        
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.tree.yview)
        scrollbar.pack(side="right", fill="y")
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        # Bind double click to play
        self.tree.bind("<Double-1>", lambda e: self.play_selected())
        
        # Control Frame
        ctrl_frame = ttk.Frame(self.root, padding="10")
        ctrl_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Button(ctrl_frame, text="Play Selected", command=self.play_selected).pack(side="left", padx=5)
        ttk.Button(ctrl_frame, text="Play All Sequentially", command=self.play_all_sequentially).pack(side="left", padx=5)
        
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        ttk.Label(ctrl_frame, textvariable=self.status_var).pack(side="right", padx=5)

    def perform_search(self):
        self.matches = []
        
        q_color = self.entry_color.get().strip()
        q_type = self.entry_type.get().strip()
        q_plate = self.entry_plate.get().strip()
        q_time = self.entry_time.get().strip()
        
        target_minutes = None
        if q_time:
            try:
                t_obj = datetime.strptime(q_time, "%H:%M")
                target_minutes = t_obj.hour * 60 + t_obj.minute
            except ValueError:
                messagebox.showerror("Error", "Invalid Time Format. Use HH:MM (24h)")
                return
        
        for vid, event in self.data.items():
            match = True
            
            if q_color:
                # Check fuzzy color match or exact hex match
                sim = string_similarity(q_color, event.get('color', ''))
                if sim < 0.6 and q_color.lower() not in event.get('color', '').lower():
                    match = False
            
            if q_type:
                if q_type.lower() not in event.get('vehicle_type', '').lower():
                    match = False
                    
            if q_plate:
                if q_plate.upper() not in event.get('plate_text', ''):
                    match = False
            
            if q_time and match:
                # Time Start-From Logic
                # Filter out clips BEFORE (Target Time - 5 mins).
                # Keep clips AFTER.
                try:
                    # first_seen format: "YYYY-MM-DD HH:MM:SS"
                    fs_str = event.get('first_seen', '')
                    if fs_str:
                        dt = datetime.strptime(fs_str, "%Y-%m-%d %H:%M:%S")
                        event_minutes = dt.hour * 60 + dt.minute
                        
                        # Threshold Minutes = Target - 5
                        threshold_minutes = target_minutes - 5
                        
                        # Handle Linear Day Logic (0..1439)
                        # If threshold wraps to previous day (e.g. 00:02 -> -3 -> 1437), strictly speaking, 
                        # "everything after 23:57 yesterday" is ambiguous without date.
                        # Assuming linear day 00:00 to 23:59.
                        # If threshold < 0, we set it to 0 (start of day) or handle wrap?
                        # User said "i only want the clips that exist BEFORE the time frame to be filetred out"
                        # For simplicity in single-day context:
                        # If event_minutes < threshold_minutes -> Filter Out
                        
                        effective_threshold = threshold_minutes
                        if effective_threshold < 0:
                             effective_threshold += 1440 # Wrap around logic? 
                             # If threshold is 23:55 (yesterday), and event is 00:05 (today).
                             # 00:05 (5) < 23:55 (1435).
                             # This would filter out 00:05.
                             # But 00:05 is "after" 23:55 if date advanced.
                             # Without date, 00:05 is "start of day".
                             # So standard comparison works fine if we assume user inputs 10:00 expecting 10:00+ events.
                             
                        # Let's stick to strict 0-1440 comparison for robust single-day behavior
                        # If target is 10:05, threshold is 10:00 (600 mins).
                        # Event 09:59 (599) < 600 -> Filtered.
                        # Event 11:00 (660) >= 600 -> Kept.
                        
                        # Handling negative threshold (e.g. input 00:02 -> threshold -3)
                        # If we treat day as circle, -3 is 23:57.
                        # But event 00:05 (5) is "after" 23:57?
                        # Or start of new day?
                        # Let's simplify: strict inequality on minutes of day.
                        # If 10:05 input. 
                        # 09:00 filtered. 12:00 kept.
                        
                        if event_minutes < threshold_minutes:
                            match = False
                            
                except Exception:
                     pass 
                    
            if match:
                self.matches.append(event)
        
        # Sort sequentially by timestamp (assuming first_seen is Sortable string YYYY-MM-DD...)
        self.matches.sort(key=lambda x: x.get('first_seen', ''))
        
        # Update UI
        for item in self.tree.get_children():
            self.tree.delete(item)
            
        for m in self.matches:
            duration = f"{m.get('last_seen_seconds', 0) - m.get('first_seen_seconds', 0):.1f}s"
            source = os.path.basename(m.get('source_video', 'Unknown'))
            self.tree.insert("", "end", values=(
                m['vehicle_id'], 
                m['vehicle_type'], 
                f"{m['color']} ({m.get('hex_value', '')})", 
                m['first_seen'], 
                duration,
                source
            ))
            
        self.status_var.set(f"Found {len(self.matches)} matches.")

    def play_clip(self, event_data):
        source = event_data.get('source_video')
        if not source or not os.path.exists(source):
            self.status_var.set(f"Error: Source video not found for {event_data['vehicle_id']}")
            return False
            
        # Create Temp File
        fd, temp_path = tempfile.mkstemp(suffix=".mp4")
        os.close(fd) # Close file descriptor, we just needed the path
        
        start = max(0, event_data.get('first_seen_seconds', 0) - 1.0)
        end = event_data.get('last_seen_seconds', 0) + 1.0
        
        self.status_var.set(f"Extracting clip for {event_data['vehicle_id']}...")
        success = extract_snippet(source, start, end, temp_path)
        
        if success:
            self.status_var.set(f"Playing {event_data['vehicle_id']}...")
            self.play_video_file(temp_path)
            
            # Cleanup
            try:
                os.remove(temp_path)
            except:
                pass
            return True
        else:
            self.status_var.set("Extraction failed.")
            return False

    def play_video_file(self, path):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened(): return
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        delay = int(1000 / fps) if fps > 0 else 30
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            cv2.imshow("Playback - Press 'q' to skip", frame)
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

    def play_selected(self):
        selected_item = self.tree.selection()
        if not selected_item: return
        
        # Get index
        index = self.tree.index(selected_item[0])
        event = self.matches[index]
        
        threading.Thread(target=self.play_clip, args=(event,)).start()

    def play_all_sequentially(self):
        if not self.matches: return
        
        def run_sequence():
            for event in self.matches:
                self.root.after(0, lambda e=event: self.tree.selection_set(self.tree.get_children()[self.matches.index(e)]))
                self.root.after(0, lambda e=event: self.tree.see(self.tree.get_children()[self.matches.index(e)]))
                success = self.play_clip(event)
                if not success: break
            self.status_var.set("Sequence finished.")
            
        threading.Thread(target=run_sequence).start()

def launch_ui():
    root = tk.Tk()
    app = VehicleSearchApp(root)
    root.mainloop()

if __name__ == "__main__":
    launch_ui()
