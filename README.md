# Crowdsafe
# CrowdSafe - Real-Time Crowd Detection using YOLOv8

A project built for the **Smart India Hackathon 2025**, CrowdSafe leverages advanced computer vision algorithms to detect and monitor crowd density in videos. The system focuses on real-time detection and analysis, ensuring public safety by classifying crowd levels as **Low**, **Moderate**, or **High** based on the number of people detected in each frame.

---

## 📌 Features

✔ Detects and counts humans in pre-recorded videos using **YOLOv8**.  
✔ Filters out irrelevant objects like bottles, cars, etc., focusing only on people.  
✔ Displays bounding boxes around detected persons in each frame.  
✔ Adds a dynamic crowd density bar at the bottom of the video.  
✔ Classifies the crowd into three levels:  
   - **Low** (green),  
   - **Moderate** (yellow),  
   - **High** (red).  
✔ Outputs the processed video with overlays and analytics.

---

## 📂 Files in the repository

- `crowd_detection.py` - Main Python script for detecting and analyzing crowds.
- `yolov8l.pt` - Pre-trained YOLOv8 model used for object detection.
- `whatsapp_video.mp4` - Sample video for testing the application.
- `output_crowd.mp4` - Output video with detected crowd annotations.
- `README.md` - This file explaining the project.

---

## ⚙ Requirements

Install dependencies using:

```bash
pip install ultralytics opencv-python numpy

## YOLOv8 Weights

This project uses the **YOLOv8 Large** model (`yolovl.pt`) for object detection.

Since the file is large (>300 MB), it is not stored directly in this repository.  
Instead, download it automatically using:

```bash
python download_weights.py
