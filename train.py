from ultralytics import YOLO

import torch

# Check Apple GPU availability

device = "mps" if torch.backends.mps.is_available() else "cpu"

print("Selected device:", device)

# Load pretrained YOLOv8 Nano model

model = YOLO("yolov8n.pt")

# Train model on VisDrone

results = model.train(

    data="VisDrone.yaml",

    epochs=1,

    imgsz=640,

    batch=4,

    device=device,

    workers=0,

    project="runs/train",

    name="visdrone_gpu_test"

)

print("Training completed successfully!")