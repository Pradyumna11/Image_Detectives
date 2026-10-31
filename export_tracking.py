from ultralytics import YOLO
from pathlib import Path
import csv
import cv2
import time

# Load trained model
model = YOLO(
    "runs/detect/runs/train/visdrone_full/weights/best.pt"
)

# Input video
video_path = Path("videos/real_video.mp4")

# Output directory
output_dir = Path("tracking_results")
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "tracking_predictions.csv"

# Open video
cap = cv2.VideoCapture(str(video_path))

if not cap.isOpened():
    raise RuntimeError("Cannot open video.")

frame_number = 0
total_predictions = 0

start_time = time.perf_counter()

with open(output_file, "w", newline="") as file:

    writer = csv.writer(file)

    writer.writerow([
        "frame",
        "track_id",
        "x",
        "y",
        "width",
        "height",
        "confidence",
        "class_id"
    ])

    while True:

        success, frame = cap.read()

        if not success:
            break

        frame_number += 1

        results = model.track(
            frame,
            tracker="bytetrack.yaml",
            persist=True,
            device="mps",
            conf=0.3,
            imgsz=640,
            verbose=False
        )

        for result in results:

            if (
                result.boxes is None
                or result.boxes.id is None
            ):
                continue

            for box in result.boxes:

                track_id = int(box.id.item())
                class_id = int(box.cls.item())
                confidence = float(box.conf.item())

                x1, y1, x2, y2 = box.xyxy[0].tolist()

                writer.writerow([
                    frame_number,
                    track_id,
                    x1,
                    y1,
                    x2 - x1,
                    y2 - y1,
                    confidence,
                    class_id
                ])

                total_predictions += 1

cap.release()

elapsed = time.perf_counter() - start_time

print("Tracking export completed!")
print("Frames processed:", frame_number)
print("Tracked detections:", total_predictions)
print("End-to-end processing FPS:", round(frame_number / elapsed, 2))
print("Saved predictions to:", output_file.resolve())