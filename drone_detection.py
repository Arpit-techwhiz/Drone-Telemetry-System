"""
Project Name: Drone Navigator's Telemetry System
Author: Arpit Chaudhary
Model Used: YOLOv8n.pt
Objective: Object Detection for Rescue Operations
"""

import cv2
import numpy as np
import time
import winsound
import csv
from datetime import datetime
from ultralytics import YOLO

# Load YOLOv8 Model
model = YOLO("yolov8n.pt")

# Create Mission Log File
log_file = "mission_log.csv"

with open(log_file, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Time", "Target ID", "Object", "Distance (m)", "Status"])

# Main Drone Feed Function
def process_drone_feed(source):

    cap = cv2.VideoCapture(source)

    prev_time = 0

    # Target ID counter
    target_id_counter = 1

    # Store locked targets (avoid repeated logging)
    locked_targets = set()

    while True:

        # Read Frame
        ret, frame = cap.read()
        if not ret:
            break

        # Mirror Fix
        frame = cv2.flip(frame, 1)

        # Resize for speed
        frame = cv2.resize(frame, (640, 640))

        # YOLO Inference
        results = model(frame)

        # Frame size
        h, w, _ = frame.shape

        # Crosshair Center
        center_screen_x = w // 2
        center_screen_y = h // 2

        cv2.drawMarker(
            frame,
            (center_screen_x, center_screen_y),
            (255, 255, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=25,
            thickness=2
        )

        # Mission HUD Text
        cv2.putText(
            frame,
            "MISSION MODE: MULTI-TARGET RESCUE SCAN",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )

        # Confidence Threshold
        threshold = 0.6

        # Lock Zone
        left_zone = w * 0.4
        right_zone = w * 0.6

        # Detection Loop
        for box in results[0].boxes:

            conf = float(box.conf[0])

            if conf > threshold:

                # Bounding Box
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Object Name
                class_id = int(box.cls[0])
                object_name = model.names[class_id]

                # Center of Object
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                # Distance Simulation
                box_height = y2 - y1
                distance = round(1000 / box_height, 2)

                # MULTI TARGET LOCK
                if left_zone < center_x < right_zone:

                    target_key = (center_x // 30, center_y // 30)

                    if target_key not in locked_targets:

                        locked_targets.add(target_key)

                        # Assign New Target ID
                        target_id = target_id_counter
                        target_id_counter += 1

                        # Beep Alert
                        winsound.Beep(1200, 250)

                        # ✅ Log Event to CSV
                        lock_time = datetime.now().strftime("%H:%M:%S")

                        with open(log_file, mode="a", newline="") as file:
                            writer = csv.writer(file)
                            writer.writerow(
                                [lock_time, target_id, object_name.upper(), distance, "LOCKED"]
                            )

                    else:
                        target_id = "LOCKED"

                    color = (0, 0, 255)
                    status = f"TARGET LOCKED #{target_id}"

                    # Tracking Line
                    cv2.line(
                        frame,
                        (center_screen_x, center_screen_y),
                        (center_x, center_y),
                        (0, 0, 255),
                        2
                    )

                else:
                    color = (0, 255, 0)
                    status = "SCANNING"

                # Draw Bounding Box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                label = f"{object_name.upper()} | {status} | Dist: {distance}m"

                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2
                )

                cv2.circle(frame, (center_x, center_y), 5, color, -1)

        # FPS Counter
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        cv2.putText(
            frame,
            f"FPS: {int(fps)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

        # Display Window
        cv2.imshow("Drone Telemetry System (Mission Logging)", frame)

        # Exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# Run Webcam Feed
process_drone_feed(0)
