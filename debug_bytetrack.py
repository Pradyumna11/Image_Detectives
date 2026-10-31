git remote -vfrom pathlib import Path
import cv2
import pandas as pd
import torch

from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / "runs/detect/runs/train/visdrone_full/weights/best.pt"
)

SEQUENCE_DIR = (
    BASE_DIR
    / "videos/visdrone_mot/VisDrone2019-MOT-val"
    / "sequences/uav0000268_05773_v"
)

OUTPUT_DIR = (
    BASE_DIR
    / "tracking_results/bytetrack_debug"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

START_FRAME = 301
END_FRAME = 400

IMAGE_SIZE = 1280
CONFIDENCE = 0.05

DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

PEDESTRIAN_CLASS = 0

CUSTOM_TRACKER = OUTPUT_DIR / "bytetrack_custom.yaml"


# ============================================================
# CREATE CUSTOM TRACKER CONFIGURATION
# ============================================================

def create_custom_tracker():

    config = """
tracker_type: bytetrack
track_high_thresh: 0.05
track_low_thresh: 0.01
new_track_thresh: 0.05
track_buffer: 30
match_thresh: 0.8
fuse_score: True
"""

    CUSTOM_TRACKER.write_text(
        config.strip() + "\n",
        encoding="utf-8"
    )

    print("Custom tracker saved:", CUSTOM_TRACKER)


# ============================================================
# LOAD FRAMES
# ============================================================

def load_frames():

    images = sorted(
        image
        for image in SEQUENCE_DIR.glob("*.jpg")
        if START_FRAME <= int(image.stem) <= END_FRAME
    )

    if not images:
        raise RuntimeError("No frames found.")

    return images


# ============================================================
# DETECTION CONFIDENCE ANALYSIS
# ============================================================

def analyze_detections(images):

    print("\n" + "=" * 60)
    print("YOLOv8 PEDESTRIAN DETECTION ANALYSIS")
    print("=" * 60)

    model = YOLO(str(MODEL_PATH))

    pedestrian_scores = []
    total_pedestrian_detections = 0

    for image_path in images:

        frame = cv2.imread(str(image_path))

        if frame is None:
            raise RuntimeError(
                f"Cannot read frame: {image_path}"
            )

        result = model.predict(
            source=frame,
            conf=CONFIDENCE,
            imgsz=IMAGE_SIZE,
            device=DEVICE,
            verbose=False
        )[0]

        boxes = result.boxes

        if boxes is None:
            continue

        classes = boxes.cls.int().cpu().tolist()
        scores = boxes.conf.cpu().tolist()

        for class_id, score in zip(classes, scores):

            if class_id == PEDESTRIAN_CLASS:

                pedestrian_scores.append(score)
                total_pedestrian_detections += 1

    print(
        "Total pedestrian detections:",
        total_pedestrian_detections
    )

    if pedestrian_scores:

        series = pd.Series(pedestrian_scores)

        print("\nPedestrian confidence statistics:")
        print(series.describe())

        print("\nConfidence distribution:")

        ranges = [
            (0.05, 0.10),
            (0.10, 0.15),
            (0.15, 0.20),
            (0.20, 0.25),
            (0.25, 1.01),
        ]

        for low, high in ranges:

            count = sum(
                low <= score < high
                for score in pedestrian_scores
            )

            print(
                f"{low:.2f} - {high:.2f}: {count}"
            )

    return total_pedestrian_detections


# ============================================================
# TRACKING EXPERIMENT
# ============================================================

def test_tracker(images, tracker_name, tracker_config):

    print("\n" + "=" * 60)
    print("TRACKER:", tracker_name)
    print("=" * 60)

    model = YOLO(str(MODEL_PATH))

    total_pedestrian_tracks = 0
    frames_with_pedestrian_tracks = 0

    unique_ids = set()

    tracker_settings_printed = False

    for index, image_path in enumerate(images, start=1):

        frame = cv2.imread(str(image_path))

        if frame is None:
            raise RuntimeError(
                f"Cannot read frame: {image_path}"
            )

        result = model.track(
            source=frame,
            tracker=str(tracker_config),
            persist=True,
            conf=CONFIDENCE,
            imgsz=IMAGE_SIZE,
            device=DEVICE,
            verbose=False
        )[0]

        # Verify the tracker configuration actually loaded.
        if not tracker_settings_printed:

            predictor = model.predictor

            if (
                predictor is not None
                and hasattr(predictor, "trackers")
                and predictor.trackers
            ):

                tracker = predictor.trackers[0]

                print("\nLoaded tracker settings:")

                for setting in [
                    "track_high_thresh",
                    "track_low_thresh",
                    "new_track_thresh",
                    "track_buffer",
                    "match_thresh",
                ]:

                    value = getattr(
                        tracker.args,
                        setting,
                        "Not available"
                    )

                    print(f"{setting}: {value}")

            else:

                print(
                    "Tracker configuration could not "
                    "be inspected."
                )

            tracker_settings_printed = True

        boxes = result.boxes

        pedestrian_tracks_this_frame = 0

        if boxes is not None and boxes.id is not None:

            classes = boxes.cls.int().cpu().tolist()
            ids = boxes.id.int().cpu().tolist()

            for class_id, track_id in zip(classes, ids):

                if class_id == PEDESTRIAN_CLASS:

                    total_pedestrian_tracks += 1
                    pedestrian_tracks_this_frame += 1

                    unique_ids.add(track_id)

        if pedestrian_tracks_this_frame > 0:

            frames_with_pedestrian_tracks += 1

        if index % 25 == 0:

            print(
                f"Processed {index}/{len(images)} frames"
            )

    return {
        "Tracker": tracker_name,
        "Image Size": IMAGE_SIZE,
        "Confidence": CONFIDENCE,
        "Frames": len(images),
        "Pedestrian Track Observations": total_pedestrian_tracks,
        "Frames With Pedestrian Tracks": frames_with_pedestrian_tracks,
        "Unique Pedestrian Track IDs": len(unique_ids),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    if not SEQUENCE_DIR.is_dir():
        raise FileNotFoundError(
            f"Sequence directory missing: {SEQUENCE_DIR}"
        )

    images = load_frames()

    create_custom_tracker()

    print("=" * 60)
    print("IMAGE DETECTIVES — BYTETRACK DEBUGGING")
    print("=" * 60)

    print("Frames:", START_FRAME, "-", END_FRAME)
    print("Image size:", IMAGE_SIZE)
    print("Confidence:", CONFIDENCE)
    print("Device:", DEVICE)

    detection_count = analyze_detections(images)

    experiments = [
        ("Default ByteTrack", "bytetrack.yaml"),
        ("Custom ByteTrack", CUSTOM_TRACKER),
    ]

    results = []

    for name, config in experiments:

        result = test_tracker(
            images,
            name,
            config
        )

        result["YOLO Pedestrian Detections"] = detection_count

        results.append(result)

        print(
            "\nPedestrian track observations:",
            result["Pedestrian Track Observations"]
        )

    results_df = pd.DataFrame(results)

    output_path = (
        OUTPUT_DIR / "bytetrack_debug_results.csv"
    )

    results_df.to_csv(
        output_path,
        index=False
    )

    print("\n" + "=" * 60)
    print("FINAL DEBUGGING RESULTS")
    print("=" * 60)

    print(results_df.to_string(index=False))

    print("\nSaved to:", output_path)


if __name__ == "__main__":
    main()
    