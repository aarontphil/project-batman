import tkinter as tk
from tkinter import ttk, messagebox
import json
import os
import cv2
import tempfile
import threading
from datetime import datetime
from tkinter import filedialog
from search_tool import extract_snippet, string_similarity, analyze_image, search_by_visuals

class VehicleSearchApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Vehicle Search & Playback")
        self.root.geometry("1000x700")
        
        self.db_path = "vehicle_db.json"
        self.data = {}
        self.matches = []
        self.is_playing = False
        
        # Load Data
        self.load_database()
        
        # Apply Theme
        self.apply_theme()
        
        # UI Components
        self.create_widgets()
        
    def apply_theme(self):
        style = ttk.Style()
        style.theme_use('clam') 
        
        # Government / Terminal Palette
        bg_root = "#121212"      # Deep Charcoal
        fg_text = "#00FF41"      # Terminal Neon Green
        fg_accent = "#FFB000"    # Amber for specific highlights
        
        self.root.configure(bg=bg_root)
        
        # General Defaults
        style.configure(".", 
            background=bg_root, 
            foreground=fg_text, 
            fieldbackground=bg_root,
            font=("Consolas", 10),
            borderwidth=1
        )
        
        # Labelframes
        style.configure("TLabelframe", 
            background=bg_root, 
            foreground=fg_text,
            bordercolor=fg_text,
            borderwidth=2
        )
        style.configure("TLabelframe.Label", 
            background=bg_root, 
            foreground=fg_text,
            font=("Consolas", 10, "bold")
        )
        
        # Buttons (Hard Edges)
        style.configure("TButton", 
            background="#222222", 
            foreground=fg_text, 
            borderwidth=1,
            focusthickness=2,
            focuscolor=fg_text,
            font=("Consolas", 10, "bold")
        )
        style.map("TButton", 
            background=[("active", "#333333"), ("pressed", "#444444")],
            foreground=[("active", fg_text)]
        )
        
        # Entries
        style.configure("TEntry", 
            fieldbackground="#000000", 
            foreground=fg_text,
            insertcolor=fg_text,
            borderwidth=1,
            relief="solid"
        )
        
        # Treeview
        style.configure("Treeview", 
            background="black",
            fieldbackground="black",
            foreground=fg_text,
            font=("Consolas", 9),
            rowheight=25,
            borderwidth=0
        )
        style.configure("Treeview.Heading", 
            background="#222222", 
            foreground=fg_text, 
            font=("Consolas", 10, "bold"),
            relief="raised"
        )
        style.map("Treeview", 
            background=[("selected", "#003300")], 
            foreground=[("selected", fg_text)]
        )
        
        # Scrollbars
        style.configure("Vertical.TScrollbar", 
            background="#222222",
            troughcolor=bg_root,
            arrowcolor=fg_text,
            borderwidth=1,
            relief="flat"
        )

    def load_database(self):
        if os.path.exists(self.db_path):
            with open(self.db_path, "r") as f:
                self.data = json.load(f)
        else:
            messagebox.showerror("Error", f"Database not found: {self.db_path}")

    def create_widgets(self):
        # Input Frame
        input_frame = ttk.LabelFrame(self.root, text="Target Parameters [SEARCH]", padding="10")
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
        
        # Routemap Button (Right Bottom)
        ttk.Button(ctrl_frame, text="Generate Routemap", command=self.generate_routemap).pack(side="right", padx=5)
        
        # Visual Search Button (Right Bottom)
        ttk.Button(ctrl_frame, text="Search by Image", command=self.browse_image).pack(side="right", padx=5)

        # Case File Button (Right Bottom)
        ttk.Button(ctrl_frame, text="Generate Case File 📄", command=self.generate_report).pack(side="right", padx=5)
        
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

    def generate_report(self):
        selected_item = self.tree.selection()
        if not selected_item:
            messagebox.showinfo("Info", "Select a vehicle to generate a report.")
            return
            
        index = self.tree.index(selected_item[0])
        event = self.matches[index]
        
        try:
            from case_file import generate_case_report
            
            self.status_var.set("Generating PDF Case File...")
            
            output_name = f"CaseFile_{event['vehicle_id']}.pdf"
            success = generate_case_report(event, output_name)
            
            if success:
                self.status_var.set(f"Report Generated: {output_name}")
                try:
                    os.startfile(output_name)
                except:
                    messagebox.showinfo("Success", f"Report saved as {output_name}")
            else:
                 self.status_var.set("Report generation failed.")
                 messagebox.showerror("Error", "Could not generate report.")
                 
        except Exception as e:
            print(e)
            self.status_var.set(f"Error: {e}")
            messagebox.showerror("Error", f"Report Error: {e}")

    def browse_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")]
        )
        if not file_path:
            return
            
        self.status_var.set("Analyzing image...")
        
        def run_analysis():
            try:
                query_event, best_box = analyze_image(file_path)
                
                if not query_event:
                    self.root.after(0, lambda: messagebox.showinfo("Info", "No vehicle detected."))
                    self.root.after(0, lambda: self.status_var.set("Analysis failed."))
                    return

                # Auto-fill Filters (Main Thread)
                def update_ui():
                    # Color
                    color = query_event.get('color', '')
                    if color:
                        self.entry_color.delete(0, tk.END)
                        self.entry_color.insert(0, color)
                    
                    # Type (Optional, but users usually want this too)
                    v_type = query_event.get('vehicle_type', '')
                    if v_type:
                        self.entry_type.delete(0, tk.END)
                        self.entry_type.insert(0, v_type)
                        
                    # Run Text Search First
                    self.perform_search()
                    
                    # Visual Re-Ranking
                    # Now we take the results from perform_search (which are filtered by color/type)
                    # and sort them by visual similarity to the query image
                    if self.matches:
                         scored = search_by_visuals(query_event, {m['vehicle_id']: m for m in self.matches}, top_k=len(self.matches))
                         # Update matches with sorted list
                         self.matches = [s[1] for s in scored]
                         self.populate_results_tree()
                         self.status_var.set(f"Found {len(self.matches)} matches (Color Filtered & Sorted).")
                    else:
                         self.status_var.set(f"No matches found for {color} {v_type}.")

                self.root.after(0, update_ui)
                
            except Exception as e:
                print(e)
                self.root.after(0, lambda: messagebox.showerror("Error", f"Analysis error: {e}"))

        threading.Thread(target=run_analysis).start()

    def generate_routemap(self):
        if not self.matches:
            messagebox.showinfo("Info", "No matches found to generate map.")
            return

        try:
            from PIL import Image, ImageTk
        except ImportError:
            messagebox.showerror("Error", "Pillow (PIL) library not found. Please install it: pip install Pillow")
            return

        # Create New Window
        map_window = tk.Toplevel(self.root)
        map_window.title(f"Route Map ({len(self.matches)} points)")
        map_window.geometry("1000x400")
        
        # Scrollable Frame
        canvas = tk.Canvas(map_window)
        scrollbar = ttk.Scrollbar(map_window, orient="horizontal", command=canvas.xview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(xscrollcommand=scrollbar.set)

        canvas.pack(side="top", fill="both", expand=True)
        scrollbar.pack(side="bottom", fill="x")

        # Populate Images
        image_refs = [] # Keep references to avoid GC
        
        for i, event in enumerate(self.matches):
            source = event.get('source_video', '')
            if not os.path.exists(source): continue
            
            # Extract Frame
            cap = cv2.VideoCapture(source)
            # Use middle of the clip or first seen? First seen is better for 'start' of sighting.
            # But converting seconds to frame index.
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0: fps = 30
            
            timestamp = event.get('first_seen_seconds', 0)
            frame_idx = int(timestamp * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Resize for thumbnail (e.g. height 200, keep aspect)
                h, w, _ = rgb_frame.shape
                target_h = 250
                aspect = w / h
                target_w = int(target_h * aspect)
                
                img_pil = Image.fromarray(rgb_frame)
                img_pil = img_pil.resize((target_w, target_h), Image.Resampling.LANCZOS)
                
                img_tk = ImageTk.PhotoImage(img_pil)
                image_refs.append(img_tk) # Hold reference
                
                # Container for this Step
                step_frame = ttk.Frame(scrollable_frame, padding="10")
                step_frame.pack(side="left", padx=5)
                
                # Image Label
                lbl_img = ttk.Label(step_frame, image=img_tk)
                lbl_img.pack()
                
                # Info Label
                time_str = event.get('first_seen', 'N/A').split(' ')[-1] # Just time part
                lbl_info = ttk.Label(step_frame, text=f"{time_str}\n{event.get('vehicle_type','')}", justify="center")
                lbl_info.pack(pady=5)
                
                # Arrow (if not last)
                if i < len(self.matches) - 1:
                    lbl_arrow = ttk.Label(scrollable_frame, text="➜", font=("Arial", 20))
                    lbl_arrow.pack(side="left", padx=5)

        # Store refs in window to prevent GC
        map_window.image_refs = image_refs

def launch_ui():
    root = tk.Tk()
    app = VehicleSearchApp(root)
    root.mainloop()

if __name__ == "__main__":
    launch_ui()
