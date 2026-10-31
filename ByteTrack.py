from ultralytics import YOLO
from pathlib import Path

# Load trained YOLOv8 model
model = YOLO(
    "runs/detect/runs/train/visdrone_full/weights/best.pt"
)

# Real tracking video

video_path = Path("videos/real_video.mp4")

if not video_path.exists():
    raise FileNotFoundError(
        f"Video not found: {video_path}"
    )

# Run ByteTrack
results = model.track(
    source=str(video_path),
    tracker="bytetrack.yaml",
    device="mps",
    conf=0.1,
    imgsz=640,
    stream=True,
    save=True,
    project="tracking_results",
    name="real_video_tracking"
)

# Count processed frames
frame_count = 0
tracked_frames = 0

for result in results:
    frame_count += 1

    if result.boxes is not None and result.boxes.id is not None:
        tracked_frames += 1

print("Total frames processed:", frame_count)
print("Frames with tracked objects:", tracked_frames)
print("Tracking completed successfully!")