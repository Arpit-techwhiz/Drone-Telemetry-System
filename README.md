# 🚁 Drone Telemetry System (YOLOv8 Target Lock)

A real-time **Drone Navigator Telemetry System** built using  
**YOLOv8 + OpenCV + NumPy**, designed for search-and-rescue style missions.

This project detects objects from a drone’s camera feed and displays a live
Heads-Up Display (HUD) with:

- Object Detection  
- Confidence Filtering  
- Target Lock System  
- FPS Counter  
- Mission Logging  

---

## 📌 Project Objective

Modern rescue drones must scan environments and automatically lock onto
important targets such as:

- Humans  
- Vehicles  
- Rescue objects  

This system uses a pre-trained **YOLOv8 neural network** to perform detection
and overlays telemetry-style visuals in real time.

---

## 🛠 Tech Stack

- **Python 3**
- **YOLOv8 (Ultralytics)**
- **OpenCV**
- **NumPy**
- **Math + Real-time Logic**

---

## ⚙️ Features Implemented

✅ Live Drone Camera / Video Feed Input  
✅ Bounding Box Detection  
✅ Confidence Threshold Filtering (removes ghost detections)  
✅ Target Lock Logic:

- **Red Box** → Target Locked (center zone)
- **Green Box** → Scanning Mode

✅ Simulated Distance Estimation  
✅ FPS Counter Display  
✅ Mission Log Output (`mission_log.csv`)

---

## 🎯 Target Lock Logic

The drone locks onto an object when its bounding box center lies in the middle
20% of the screen width:

- If object center is inside lock zone → **TARGET LOCKED**
- Otherwise → **SCANNING**

This mimics real autonomous drone HUD behavior.

---

## AUTHOR
*Arpit Chaudhary*

