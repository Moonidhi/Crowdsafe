from ultralytics import YOLO
import cv2
import numpy as np
import csv
import datetime

# Load YOLO model
model = YOLO("yolov8l.pt")  # Download this model file and place it in the same folder

# Video file path
video_path = "WhatsApp Video 2025-09-13 at 15.47.40.mp4"

# Open video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Cannot open video")
    exit()

# CSV file to log data
csv_file = "crowd_data.csv"
with open(csv_file, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Timestamp", "People Count", "Crowd Level"])

# Crowd thresholds
LOW = 5
MODERATE = 15

def get_crowd_level(count):
    if count <= LOW:
        return "Low", (0, 255, 0)
    elif count <= MODERATE:
        return "Moderate", (0, 255, 255)
    else:
        return "High", (0, 0, 255)

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video")
        break

    frame = cv2.resize(frame, (640, 480))

    # Detect humans
    results = model.predict(frame, imgsz=640, conf=0.3)
    person_count = 0

    for result in results:
        boxes = result.boxes
        for box, cls, conf in zip(boxes.xyxy, boxes.cls, boxes.conf):
            class_id = int(cls)
            confidence = float(conf)
            if class_id == 0 and confidence > 0.3:  # Only humans
                x1, y1, x2, y2 = map(int, box)
                width = x2 - x1
                height = y2 - y1
                if width > 20 and height > 20:  # Filter tiny boxes like hands
                    person_count += 1
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"Person {confidence:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show people count
    cv2.putText(frame, f"People Count: {person_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Crowd level bar
    crowd_level, color = get_crowd_level(person_count)
    cv2.rectangle(frame, (0, 450), (640, 480), color, -1)
    cv2.putText(frame, f"Crowd Level: {crowd_level}", (200, 470),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # Show frame
    cv2.imshow("Crowd Detection", frame)

    # Log data
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(csv_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, person_count, crowd_level])

    # Quit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
