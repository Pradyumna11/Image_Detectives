from pathlib import Path
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

SEQUENCE_NAME = "uav0000268_05773_v"

DATASET_DIR = (
    BASE_DIR
    / "videos/visdrone_mot/VisDrone2019-MOT-val"
)

SEQUENCE_DIR = (
    DATASET_DIR / "sequences" / SEQUENCE_NAME
)

ANNOTATION_PATH = (
    DATASET_DIR
    / "annotations"
    / f"{SEQUENCE_NAME}.txt"
)

OUTPUT_DIR = (
    BASE_DIR
    / "tracking_results"
    / "diagnostics"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

START_FRAME = 301
END_FRAME = 400

CONFIDENCE_LEVELS = [0.05, 0.10, 0.20, 0.30]

IMAGE_SIZE = 640

DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

PEDESTRIAN_CLASS = 0


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
            f"Sequence directory not found: {SEQUENCE_DIR}"
        )

    if not ANNOTATION_PATH.is_file():
        raise FileNotFoundError(
            f"Annotations not found: {ANNOTATION_PATH}"
        )

    images = sorted(
        image
        for image in SEQUENCE_DIR.glob("*.jpg")
        if START_FRAME <= int(image.stem) <= END_FRAME
    )

    if not images:
        raise RuntimeError(
            "No images found in selected frame range."
        )

    gt = pd.read_csv(
        ANNOTATION_PATH,
        header=None
    )

    gt_pedestrians = gt[
        (gt[0].between(START_FRAME, END_FRAME))
        & (gt[7] == 1)
        & (gt[6] == 1)
        & (gt[1] > 0)
    ]

    print("=" * 65)
    print("IMAGE DETECTIVES — DETECTION VS TRACKING DIAGNOSTIC")
    print("=" * 65)

    print("Sequence:", SEQUENCE_NAME)
    print("Frames:", START_FRAME, "-", END_FRAME)
    print("Ground-truth pedestrian observations:", len(gt_pedestrians))
    print("Device:", DEVICE)

    results_summary = []

    for confidence in CONFIDENCE_LEVELS:

        print(f"\nTesting confidence: {confidence:.2f}")

        # Separate model instances
        detection_model = YOLO(str(MODEL_PATH))
        tracking_model = YOLO(str(MODEL_PATH))

        raw_pedestrian_detections = 0
        tracked_pedestrian_observations = 0
        frames_with_detections = 0
        frames_with_tracks = 0

        for index, image_path in enumerate(images, start=1):

            # --------------------------------------------
            # INDEPENDENT YOLO DETECTION
            # --------------------------------------------

            detection_results = detection_model.predict(
                source=str(image_path),
                conf=confidence,
                imgsz=IMAGE_SIZE,
                device=DEVICE,
                verbose=False
            )

            detection_boxes = detection_results[0].boxes

            if detection_boxes is not None:

                detection_classes = (
                    detection_boxes.cls.int().cpu().tolist()
                )

                pedestrian_count = detection_classes.count(
                    PEDESTRIAN_CLASS
                )

                raw_pedestrian_detections += pedestrian_count

                if pedestrian_count > 0:
                    frames_with_detections += 1

            # --------------------------------------------
            # BYTETRACK
            # --------------------------------------------

            tracking_results = tracking_model.track(
                source=str(image_path),
                tracker="bytetrack.yaml",
                persist=True,
                conf=confidence,
                imgsz=IMAGE_SIZE,
                device=DEVICE,
                verbose=False
            )

            tracking_boxes = tracking_results[0].boxes

            if (
                tracking_boxes is not None
                and tracking_boxes.id is not None
            ):

                tracking_classes = (
                    tracking_boxes.cls.int().cpu().tolist()
                )

                tracked_count = tracking_classes.count(
                    PEDESTRIAN_CLASS
                )

                tracked_pedestrian_observations += tracked_count

                if tracked_count > 0:
                    frames_with_tracks += 1

            if index % 25 == 0:
                print(
                    f"  Processed {index}/{len(images)} frames"
                )

        results_summary.append(
            {
                "Confidence": confidence,
                "Frames": len(images),
                "GT Pedestrian Observations": len(gt_pedestrians),
                "YOLO Pedestrian Detections": raw_pedestrian_detections,
                "Frames With YOLO Detections": frames_with_detections,
                "ByteTrack Pedestrian Observations": tracked_pedestrian_observations,
                "Frames With Pedestrian Tracks": frames_with_tracks,
            }
        )

        print(
            "YOLO pedestrian detections:",
            raw_pedestrian_detections
        )

        print(
            "ByteTrack pedestrian observations:",
            tracked_pedestrian_observations
        )

    results_df = pd.DataFrame(results_summary)

    output_path = (
        OUTPUT_DIR
        / "detection_vs_tracking.csv"
    )

    results_df.to_csv(
        output_path,
        index=False
    )

    print("\n" + "=" * 65)
    print("FINAL DIAGNOSTIC RESULTS")
    print("=" * 65)

    print(
        results_df.to_string(index=False)
    )

    print("\nSaved to:", output_path)


if __name__ == "__main__":
    main()