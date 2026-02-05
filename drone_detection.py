"""
Project Name: Drone Navigator Telemetry System
Author: Arpit Chaudhary
Model Used: YOLOv8n.pt

Objective:
Real-time object detection from drone feed with Target Lock HUD and Mission Logging.
"""

import cv2
import numpy as np
import time
import csv
from datetime import datetime
from ultralytics import YOLO

# Load YOLOv8 Model
model = YOLO("yolov8n.pt")

# Main Function
def process_drone_feed(source=0):

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(" Error: Cannot open video source!")
        return

    # Frame width (for Target Lock zone calculation)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    # Target Lock Zone (Middle 20%)
    lock_left = int(frame_width * 0.40)
    lock_right = int(frame_width * 0.60)

    # FPS calculation
    prev_time = 0

    # Frame counter for logging control
    frame_count = 0

    # Mission Log File Setup
    log_file = open("mission_log.csv", mode="w", newline="")
    log_writer = csv.writer(log_file)

    # Header row
    log_writer.writerow([
        "Timestamp",
        "Object",
        "Confidence",
        "Status",
        "CenterX",
        "CenterY",
        "SimulatedDistance"
    ])

    print(" Mission Started... Logging Enabled!")
    # Main Video Loop
    while True:
        ret, frame = cap.read()
        if not ret:
            print(" Video feed ended.")
            break

        frame_count += 1

        # Resize for faster inference (Real-time Ready)
        frame = cv2.resize(frame, (640, 640))

        # YOLO Inference
        results = model(frame, stream=True)

        # Process Detections
        for result in results:
            boxes = result.boxes

            for box in boxes:
                confidence = float(box.conf[0])

                # Confidence Filtering
                if confidence < 0.6:
                    continue

                # Bounding Box Coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Class Name
                class_id = int(box.cls[0])
                class_name = model.names[class_id]

                # Center Point Calculation (NumPy Logic)
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                # Simulated Distance (Smaller box = farther)
                box_height = y2 - y1
                distance = round(1000 / (box_height + 1), 2)

                # Target Lock Logic
                if lock_left <= center_x <= lock_right:
                    color = (0, 0, 255)  # Red
                    status = "TARGET LOCKED"
                else:
                    color = (0, 255, 0)  # Green
                    status = "SCANNING"

                # Draw Bounding Box + HUD
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                label = f"{class_name} ({confidence:.2f})"
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                cv2.putText(frame, f"Dist: {distance}m", (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                cv2.putText(frame, status, (x1, y2 + 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                # Draw center point crosshair
                cv2.drawMarker(frame, (center_x, center_y),
                               color, markerType=cv2.MARKER_CROSS,
                               markerSize=15, thickness=2)

                # Mission Log Update (Every 10 frames)
                if frame_count % 10 == 0:
                    timestamp = datetime.now().strftime("%H:%M:%S")

                    log_writer.writerow([
                        timestamp,
                        class_name,
                        round(confidence, 2),
                        status,
                        center_x,
                        center_y,
                        distance
                    ])

        # FPS Counter
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
        prev_time = current_time

        cv2.putText(frame, f"FPS: {int(fps)}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        # Target Lock Zone Lines
        cv2.line(frame, (lock_left, 0), (lock_left, 640), (255, 255, 255), 1)
        cv2.line(frame, (lock_right, 0), (lock_right, 640), (255, 255, 255), 1)

        # Display Drone HUD Window
        cv2.imshow(" Drone Telemetry HUD", frame)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print(" Mission Aborted by User.")
            break
    # Cleanup Resources
    cap.release()
    cv2.destroyAllWindows()
    log_file.close()

    print(" Mission Completed. Log saved as mission_log.csv")


# Run Program
process_drone_feed(0)
