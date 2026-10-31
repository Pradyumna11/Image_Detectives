from ultralytics import YOLO
import torch

# Check Apple GPU availability
device = "mps" if torch.backends.mps.is_available() else "cpu"

print("Training device:", device)

# Load pretrained YOLOv8 Nano model
model = YOLO("yolov8n.pt")

# Train model on VisDrone
results = model.train(
    data="VisDrone.yaml",
    epochs=30,
    imgsz=640,
    batch=4,
    device=device,
    workers=0,
    patience=10,
    max_det=902,
    save=True,
    plots=True,
    project="runs/train",
    name="visdrone_full"
)

print("Full YOLOv8 training completed successfully!")