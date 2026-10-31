import cv2
from pathlib import Path

# VisDrone-MOT dataset location
dataset_path = Path(
    "videos/visdrone_mot/VisDrone2019-MOT-val/sequences"
)

# Find available sequences
sequences = sorted(
    folder for folder in dataset_path.iterdir()
    if folder.is_dir()
)

if not sequences:
    raise FileNotFoundError("No VisDrone sequences found!")

# Select the first available sequence
sequence = sequences[0]

print("Selected sequence:", sequence.name)

# Load consecutive frames in order
images = sorted(sequence.glob("*.jpg"))

if not images:
    raise FileNotFoundError("No JPG frames found!")

# Use up to 300 consecutive frames
images = images[:300]

# Read the first frame to determine dimensions
first_frame = cv2.imread(str(images[0]))

if first_frame is None:
    raise RuntimeError("Could not read first frame.")

height, width = first_frame.shape[:2]

# Output video
output_path = Path("videos/real_video.mp4")

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

video = cv2.VideoWriter(
    str(output_path),
    fourcc,
    20,
    (width, height)
)

if not video.isOpened():
    raise RuntimeError("Unable to create output video.")

# Convert consecutive images into video
for image_path in images:
    frame = cv2.imread(str(image_path))

    if frame is None:
        print("Skipping unreadable frame:", image_path)
        continue

    if frame.shape[:2] != (height, width):
        frame = cv2.resize(frame, (width, height))

    video.write(frame)

video.release()

print("Video created successfully!")
print("Sequence:", sequence.name)
print("Frames:", len(images))
print("Saved to:", output_path.resolve())