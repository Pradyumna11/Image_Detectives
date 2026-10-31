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

SEQUENCE_DIR = (
    BASE_DIR
    / "videos/visdrone_mot/VisDrone2019-MOT-val"
    / "sequences/uav0000268_05773_v"
)

OUTPUT_DIR = (
    BASE_DIR
    / "tracking_results/pedestrian_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CONFIDENCE_LEVELS = [0.05, 0.10, 0.20, 0.30]

IMAGE_SIZE = 640

DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# Analyze a sample of consecutive frames first.
MAX_FRAMES = 100

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

    images = sorted(
        SEQUENCE_DIR.glob("*.jpg")
    )[:MAX_FRAMES]

    if not images:
        raise RuntimeError(
            "No images found in the sequence."
        )

    model = YOLO(str(MODEL_PATH))

    print("=" * 60)
    print("IMAGE DETECTIVES — PEDESTRIAN DETECTION ANALYSIS")
    print("=" * 60)

    print("Sequence:", SEQUENCE_DIR.name)
    print("Frames analyzed:", len(images))
    print("Device:", DEVICE)
    print()

    results_summary = []

    for confidence in CONFIDENCE_LEVELS:

        pedestrian_detections = 0
        frames_with_pedestrians = 0
        all_class_detections = 0

        for image_path in images:

            results = model.predict(
                source=str(image_path),
                conf=confidence,
                imgsz=IMAGE_SIZE,
                device=DEVICE,
                verbose=False
            )

            boxes = results[0].boxes

            if boxes is None:
                continue

            class_ids = (
                boxes.cls.int().cpu().tolist()
            )

            pedestrian_count = class_ids.count(
                PEDESTRIAN_CLASS
            )

            pedestrian_detections += pedestrian_count
            all_class_detections += len(class_ids)

            if pedestrian_count > 0:
                frames_with_pedestrians += 1

        results_summary.append(
            {
                "Confidence": confidence,
                "Frames Analyzed": len(images),
                "Pedestrian Detections": pedestrian_detections,
                "Frames With Pedestrians": frames_with_pedestrians,
                "All-Class Detections": all_class_detections
            }
        )

        print(
            f"Confidence {confidence:.2f}: "
            f"{pedestrian_detections} pedestrian detections "
            f"in {frames_with_pedestrians} frames"
        )

    summary_df = pd.DataFrame(results_summary)

    output_path = (
        OUTPUT_DIR / "confidence_analysis.csv"
    )

    summary_df.to_csv(
        output_path,
        index=False
    )

    print("\n" + "=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)

    print(summary_df.to_string(index=False))

    print("\nSaved to:", output_path)


if __name__ == "__main__":
    main()