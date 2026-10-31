from ultralytics import YOLO

# Load pretrained YOLOv8 model
model = YOLO("yolov8n.pt")

# Perform object detection
results = model.predict(
    source="https://ultralytics.com/images/bus.jpg",
    save=True,
    conf=0.25
)

print("Object detection completed!")