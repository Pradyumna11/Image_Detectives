from ultralytics import YOLO
from pathlib import Path

# Load trained model
model = YOLO(
    "runs/detect/runs/train/visdrone_gpu_test/weights/best.pt"
)

# Locate validation images
image_folder = Path("datasets/VisDrone/images/val")

images = sorted(image_folder.glob("*.jpg"))

if not images:
    raise FileNotFoundError("No validation images found.")

# Select first image
image_path = images[0]

print("Testing image:", image_path)

# Run object detection
results = model.predict(
    source=str(image_path),
    conf=0.20,
    imgsz=640,
    device="mps",
    save=True,
    project="runs/predict",
    name="visdrone_test"
)

# Display detected objects
for result in results:
    for box in result.boxes:

        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        class_name = model.names[class_id]

        print(
            f"Detected: {class_name}, "
            f"Confidence: {confidence:.2f}"
        )

print("Object detection completed successfully!")