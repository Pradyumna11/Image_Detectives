from pathlib import Path
import time

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
    / "tracking_results/resolution_experiments"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

START_FRAME = 301
END_FRAME = 400

IMAGE_SIZES = [640, 960, 1280]

CONFIDENCE = 0.05

DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

PEDESTRIAN_CLASS = 0


# ============================================================
# RUN EXPERIMENT
# ============================================================

def run_experiment(images, image_size):

    print(f"\nTesting image size: {image_size}")

    detection_model = YOLO(str(MODEL_PATH))

    tracking_model = YOLO(str(MODEL_PATH))

    pedestrian_detections = 0
    tracked_pedestrians = 0

    frames_with_detections = 0
    frames_with_tracks = 0

    unique_track_ids = set()

    start_time = time.perf_counter()

    for index, image_path in enumerate(images, start=1):

        # ----------------------------------------------------
        # INDEPENDENT YOLO DETECTION
        # ----------------------------------------------------

        detection_result = detection_model.predict(
            source=str(image_path),
            conf=CONFIDENCE,
            imgsz=image_size,
            device=DEVICE,
            verbose=False,
        )[0]

        detection_boxes = detection_result.boxes

        if detection_boxes is not None:

            classes = (
                detection_boxes.cls.int().cpu().tolist()
            )

            count = classes.count(PEDESTRIAN_CLASS)

            pedestrian_detections += count

            if count > 0:
                frames_with_detections += 1

        # ----------------------------------------------------
        # BYTETRACK
        # ----------------------------------------------------

        tracking_result = tracking_model.track(
            source=str(image_path),
            tracker="bytetrack.yaml",
            persist=True,
            conf=CONFIDENCE,
            imgsz=image_size,
            device=DEVICE,
            verbose=False,
        )[0]

        tracking_boxes = tracking_result.boxes

        if (
            tracking_boxes is not None
            and tracking_boxes.id is not None
        ):

            track_classes = (
                tracking_boxes.cls.int().cpu().tolist()
            )

            track_ids = (
                tracking_boxes.id.int().cpu().tolist()
            )

            current_tracks = 0

            for class_id, track_id in zip(
                track_classes,
                track_ids,
            ):

                if class_id == PEDESTRIAN_CLASS:

                    tracked_pedestrians += 1
                    current_tracks += 1

                    unique_track_ids.add(track_id)

            if current_tracks > 0:
                frames_with_tracks += 1

        if index % 25 == 0:

            print(
                f"Processed {index}/{len(images)} frames"
            )

    elapsed = time.perf_counter() - start_time

    fps = (
        len(images) / elapsed
        if elapsed > 0
        else 0
    )

    return {
        "Image Size": image_size,
        "Confidence": CONFIDENCE,
        "Frames": len(images),
        "Pedestrian Detections": pedestrian_detections,
        "Frames With Detections": frames_with_detections,
        "Tracked Pedestrian Observations": tracked_pedestrians,
        "Frames With Tracks": frames_with_tracks,
        "Distinct Pedestrian Track IDs": len(unique_track_ids),
        "Combined Detection and Tracking FPS": round(fps, 2),
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
            f"Sequence not found: {SEQUENCE_DIR}"
        )

    images = sorted(
        image
        for image in SEQUENCE_DIR.glob("*.jpg")
        if START_FRAME <= int(image.stem) <= END_FRAME
    )

    if not images:
        raise RuntimeError(
            "No images found for evaluation."
        )

    print("=" * 65)
    print("IMAGE DETECTIVES — IMAGE RESOLUTION EXPERIMENT")
    print("=" * 65)

    print("Frames:", START_FRAME, "-", END_FRAME)
    print("Confidence:", CONFIDENCE)
    print("Device:", DEVICE)

    results = []

    for image_size in IMAGE_SIZES:

        experiment = run_experiment(
            images,
            image_size,
        )

        results.append(experiment)

        pd.DataFrame(results).to_csv(
            OUTPUT_DIR / "resolution_comparison.csv",
            index=False,
        )

        print(
            f"Image Size {image_size}: "
            f"{experiment['Pedestrian Detections']} detections, "
            f"{experiment['Tracked Pedestrian Observations']} "
            f"tracked observations"
        )

    results_df = pd.DataFrame(results)

    print("\n" + "=" * 65)
    print("FINAL RESOLUTION COMPARISON")
    print("=" * 65)

    print(results_df.to_string(index=False))

    print(
        "\nResults saved to:",
        OUTPUT_DIR / "resolution_comparison.csv",
    )


if __name__ == "__main__":
    main()
    