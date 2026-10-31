import cv2
from pathlib import Path

# Path to existing VisDrone images
image_folder = Path("datasets/VisDrone/images/val")

# Output video location
output_path = Path("videos/test_video.mp4")

# Create videos folder if necessary
output_path.parent.mkdir(parents=True, exist_ok=True)

# Select 100 images
images = sorted(image_folder.glob("*.jpg"))[:100]

if not images:
    raise FileNotFoundError("No images found!")

# Video settings
width = 640
height = 480
fps = 10

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

video = cv2.VideoWriter(
    str(output_path),
    fourcc,
    fps,
    (width, height)
)

if not video.isOpened():
    raise RuntimeError("Unable to create video")

# Convert images into video frames
for image_path in images:
    frame = cv2.imread(str(image_path))

    if frame is None:
        continue

    frame = cv2.resize(frame, (width, height))
    video.write(frame)

video.release()

print("Test video created successfully!")
print("Video saved at:", output_path.resolve())
