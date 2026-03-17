import tkinter as tk
import os
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO
import threading
import time

model = YOLO("yolov8m.pt")

vehicle_classes = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}

videos = [f for f in os.listdir(".") if f.endswith((".mp4", ".avi", ".mov", ".mkv"))]
videos.sort()
print(f"Found {len(videos)} videos: {videos}")

class TrafficSenseApp:
    def __init__(self, root):
        self.root = root
        self.root.title("TrafficSense - Real Time Vehicle Detection")
        self.root.configure(bg="#1a1a2e")
        self.root.state('zoomed')
        self.running = False
        self.cap = None
        self.build_ui()

    def build_ui(self):
        # Title
        tk.Label(self.root, text="TrafficSense: Real-Time Vehicle Detection & Signal Control",
                 font=("Arial", 14, "bold"), bg="#16213e", fg="#00d4ff", pady=8).pack(fill="x")

        # Buttons at TOP so they're always visible
        btn_frame = tk.Frame(self.root, bg="#1a1a2e", pady=8)
        btn_frame.pack(fill="x")

        for i, video in enumerate(videos):
            tk.Button(btn_frame, text=f"▶ Video {i+1}", font=("Arial", 11, "bold"),
                      bg="#0f3460", fg="white", padx=15, pady=6,
                      command=lambda v=i: self.start_video(v)).pack(side="left", padx=10)
        tk.Button(btn_frame, text="Stop", font=("Arial", 11, "bold"),
                  bg="#c0392b", fg="white", padx=15, pady=6,
                  command=self.stop_video).pack(side="right", padx=10)
        # Main content area
        content = tk.Frame(self.root, bg="#1a1a2e")
        content.pack(fill="both", expand=True, padx=10, pady=5)

        # Left - video
        left = tk.Frame(content, bg="#1a1a2e")
        left.pack(side="left", fill="both", expand=True)

        tk.Label(left, text="Live Video Feed", font=("Arial", 11, "bold"),
                 bg="#1a1a2e", fg="#00d4ff").pack()

        self.video_label = tk.Label(left, bg="#0f3460")
        self.video_label.pack(fill="both", expand=True, padx=5, pady=5)

        # Lane labels
        lane_frame = tk.Frame(left, bg="#1a1a2e")
        lane_frame.pack(fill="x")
        tk.Label(lane_frame, text="◀ LEFT LANE", font=("Arial", 10, "bold"),
                 bg="#1a1a2e", fg="#00aaff").pack(side="left", padx=20)
        tk.Label(lane_frame, text="RIGHT LANE ▶", font=("Arial", 10, "bold"),
                 bg="#1a1a2e", fg="#ff4444").pack(side="right", padx=20)

        # Right - stats
        right = tk.Frame(content, bg="#16213e", width=280)
        right.pack(side="right", fill="y", padx=10)
        right.pack_propagate(False)

        tk.Label(right, text="Detection Stats", font=("Arial", 12, "bold"),
                 bg="#16213e", fg="#00d4ff", pady=8).pack()

        tk.Label(right, text="Total Vehicles", font=("Arial", 10),
                 bg="#16213e", fg="#aaaaaa").pack()
        self.total_var = tk.StringVar(value="0")
        tk.Label(right, textvariable=self.total_var, font=("Arial", 32, "bold"),
                 bg="#16213e", fg="#ffffff").pack()

        tk.Frame(right, bg="#444444", height=1).pack(fill="x", pady=8)

        tk.Label(right, text="Left Lane", font=("Arial", 10, "bold"),
                 bg="#16213e", fg="#00aaff").pack()
        self.left_var = tk.StringVar(value="Vehicles: 0")
        tk.Label(right, textvariable=self.left_var, font=("Arial", 10),
                 bg="#16213e", fg="#ffffff").pack()
        self.left_density_var = tk.StringVar(value="Density: -")
        tk.Label(right, textvariable=self.left_density_var, font=("Arial", 10),
                 bg="#16213e", fg="#ffffff").pack()
        self.left_signal_var = tk.StringVar(value="Green Light: -")
        tk.Label(right, textvariable=self.left_signal_var, font=("Arial", 10),
                 bg="#16213e", fg="#00ff00").pack()

        tk.Frame(right, bg="#444444", height=1).pack(fill="x", pady=8)

        tk.Label(right, text="Right Lane", font=("Arial", 10, "bold"),
                 bg="#16213e", fg="#ff4444").pack()
        self.right_var = tk.StringVar(value="Vehicles: 0")
        tk.Label(right, textvariable=self.right_var, font=("Arial", 10),
                 bg="#16213e", fg="#ffffff").pack()
        self.right_density_var = tk.StringVar(value="Density: -")
        tk.Label(right, textvariable=self.right_density_var, font=("Arial", 10),
                 bg="#16213e", fg="#ffffff").pack()
        self.right_signal_var = tk.StringVar(value="Green Light: -")
        tk.Label(right, textvariable=self.right_signal_var, font=("Arial", 10),
                 bg="#16213e", fg="#00ff00").pack()

        tk.Frame(right, bg="#444444", height=1).pack(fill="x", pady=8)

        tk.Label(right, text="Signal Timer", font=("Arial", 10, "bold"),
                 bg="#16213e", fg="#00d4ff").pack()
        self.timer_var = tk.StringVar(value="--")
        tk.Label(right, textvariable=self.timer_var, font=("Arial", 36, "bold"),
                 bg="#16213e", fg="#ffcc00").pack()

        tk.Frame(right, bg="#444444", height=1).pack(fill="x", pady=8)

        tk.Label(right, text="Vehicle Breakdown", font=("Arial", 10, "bold"),
                 bg="#16213e", fg="#00d4ff").pack()
        self.breakdown_var = tk.StringVar(value="")
        tk.Label(right, textvariable=self.breakdown_var, font=("Arial", 10),
                 bg="#16213e", fg="#cccccc", justify="left").pack()

    def get_density(self, count):
        if count <= 5:
            return "Low", 15
        elif count <= 15:
            return "Medium", 30
        else:
            return "High", 45

    def start_video(self, video_index):
        self.stop_video()
        time.sleep(0.2)
        self.running = True
        thread = threading.Thread(target=self.process_video, args=(videos[video_index],))
        thread.daemon = True
        thread.start()

    def stop_video(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.timer_var.set("--")
        self.total_var.set("0")

    def process_video(self, video_path):
        self.cap = cv2.VideoCapture(video_path)

        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            frame = cv2.resize(frame, (800, 500))
            h, w = frame.shape[:2]
            cv2.line(frame, (w//2, 0), (w//2, h), (255, 255, 0), 2)

            results = model(frame, verbose=False)
            left_count = right_count = 0
            emergency_detected = False
            breakdown = {"Car": 0, "Motorcycle": 0, "Bus": 0, "Truck": 0}

            for result in results:
                for box in result.boxes:
                    cls = int(box.cls)
                    if cls in vehicle_classes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cx = (x1 + x2) // 2
                        color = (0, 100, 255) if cx < w//2 else (0, 0, 255)
                        if cx < w//2:
                            left_count += 1
                        else:
                            right_count += 1
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, vehicle_classes[cls], (x1, y1-5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
                        breakdown[vehicle_classes[cls]] += 1
                        # Detect emergency vehicle by size (large vehicles in center of frame)
                        box_width = x2 - x1
                        box_height = y2 - y1
                        if box_width > 150 and box_height > 150 and vehicle_classes[cls] in ["Bus", "Truck"]: 
                            emergency_detected = True
                            cv2.putText(frame, "EMERGENCY!", (x1, y1-20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            total = left_count + right_count
            left_density, left_time = self.get_density(left_count)
            right_density, right_time = self.get_density(right_count)
            signal_time = max(left_time, right_time)

            self.total_var.set(str(total))
            self.left_var.set(f"Vehicles: {left_count}")
            self.left_density_var.set(f"Density: {left_density}")
            self.left_signal_var.set(f"Green Light: {left_time}s")
            self.right_var.set(f"Vehicles: {right_count}")
            self.right_density_var.set(f"Density: {right_density}")
            self.right_signal_var.set(f"Green Light: {right_time}s")
            self.breakdown_var.set(
                f"Cars: {breakdown['Car']}\n"
                f"Motorcycles: {breakdown['Motorcycle']}\n"
                f"Buses: {breakdown['Bus']}\n"
                f"Trucks: {breakdown['Truck']}" 
            )
            if emergency_detected:
                self.timer_var.set("EMERGENCY!")
                self.timer_var_label = self.timer_var
                signal_time = 60
            else:
                self.timer_var.set(f"{signal_time}s")

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

            time.sleep(0.03)

        self.cap.release()

root = tk.Tk()
app = TrafficSenseApp(root)
root.mainloop()