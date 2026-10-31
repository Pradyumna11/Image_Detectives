from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=float: np.asarray(a, dtype=dtype)

import motmetrics as mm
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
    / "tracking_results/bytetrack_optimization"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CUSTOM_TRACKER_PATH = (
    OUTPUT_DIR / "bytetrack_low_conf.yaml"
)

CONFIDENCE = 0.05

IMAGE_SIZE = 640

DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

PROCESS_START = 1
PROCESS_END = 600

EVAL_START = 301
EVAL_END = 600

GT_COLUMNS = [
    "frame",
    "track_id",
    "x",
    "y",
    "width",
    "height",
    "score",
    "class_id",
    "truncation",
    "occlusion",
]

PRED_COLUMNS = [
    "frame",
    "track_id",
    "x",
    "y",
    "width",
    "height",
    "confidence",
    "class_id",
]


# ============================================================
# CREATE CUSTOM TRACKER
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

    CUSTOM_TRACKER_PATH.write_text(
        config.strip() + "\n"
    )

    print(
        "Custom tracker created:",
        CUSTOM_TRACKER_PATH
    )


# ============================================================
# EVALUATE PEDESTRIAN TRACKING
# ============================================================

def evaluate_tracking(gt, pred):

    gt = gt[
        gt["frame"].between(EVAL_START, EVAL_END)
    ]

    pred = pred[
        pred["frame"].between(EVAL_START, EVAL_END)
    ]

    gt = gt[
        (gt["class_id"] == 1)
        & (gt["score"] == 1)
        & (gt["track_id"] > 0)
    ]

    pred = pred[
        pred["class_id"] == 0
    ]

    gt_groups = {
        frame: group
        for frame, group in gt.groupby("frame")
    }

    pred_groups = {
        frame: group
        for frame, group in pred.groupby("frame")
    }

    accumulator = mm.MOTAccumulator(
        auto_id=True
    )

    for frame in range(EVAL_START, EVAL_END + 1):

        frame_gt = gt_groups.get(
            frame,
            gt.iloc[0:0]
        )

        frame_pred = pred_groups.get(
            frame,
            pred.iloc[0:0]
        )

        gt_ids = (
            frame_gt["track_id"]
            .astype(int)
            .tolist()
        )

        pred_ids = (
            frame_pred["track_id"]
            .astype(int)
            .tolist()
        )

        gt_boxes = frame_gt[
            ["x", "y", "width", "height"]
        ].to_numpy(dtype=float).reshape(-1, 4)

        pred_boxes = frame_pred[
            ["x", "y", "width", "height"]
        ].to_numpy(dtype=float).reshape(-1, 4)

        distances = mm.distances.iou_matrix(
            gt_boxes,
            pred_boxes,
            max_iou=0.5
        )

        accumulator.update(
            gt_ids,
            pred_ids,
            distances
        )

    handler = mm.metrics.create()

    summary = handler.compute(
        accumulator,
        metrics=[
            "mota",
            "idf1",
            "num_switches",
            "num_false_positives",
            "num_misses",
            "num_objects",
        ],
        name="tracking"
    )

    return summary.iloc[0]


# ============================================================
# RUN TRACKING EXPERIMENT
# ============================================================

def run_experiment(
    images,
    tracker_name,
    tracker_config,
    ground_truth
):

    print("\n" + "=" * 60)
    print("Tracker:", tracker_name)
    print("=" * 60)

    # Fresh model for every experiment
    model = YOLO(str(MODEL_PATH))

    predictions = []

    start_time = time.perf_counter()

    for index, image_path in enumerate(images, start=1):

        frame = int(image_path.stem)

        results = model.track(
            source=str(image_path),
            tracker=str(tracker_config),
            persist=True,
            conf=CONFIDENCE,
            imgsz=IMAGE_SIZE,
            device=DEVICE,
            verbose=False
        )

        boxes = results[0].boxes

        if boxes is not None and boxes.id is not None:

            coords = boxes.xyxy.cpu().numpy()

            ids = boxes.id.int().cpu().tolist()

            classes = (
                boxes.cls.int().cpu().tolist()
            )

            scores = boxes.conf.cpu().tolist()

            for box, track_id, class_id, score in zip(
                coords,
                ids,
                classes,
                scores
            ):

                if EVAL_START <= frame <= EVAL_END:

                    x1, y1, x2, y2 = box

                    predictions.append([
                        frame,
                        track_id,
                        float(x1),
                        float(y1),
                        float(x2 - x1),
                        float(y2 - y1),
                        float(score),
                        class_id,
                    ])

        if index % 100 == 0:

            print(
                f"Processed {index}/{len(images)} frames"
            )

    elapsed = time.perf_counter() - start_time

    fps = (
        len(images) / elapsed
        if elapsed > 0
        else 0
    )

    predictions_df = pd.DataFrame(
        predictions,
        columns=PRED_COLUMNS
    )

    predictions_df.to_csv(
        OUTPUT_DIR / f"{tracker_name}_predictions.csv",
        index=False
    )

    pedestrian_predictions = predictions_df[
        predictions_df["class_id"] == 0
    ]

    metrics = evaluate_tracking(
        ground_truth,
        predictions_df
    )

    result = {
        "Tracker": tracker_name,
        "Confidence": CONFIDENCE,
        "Pedestrian Observations": len(
            pedestrian_predictions
        ),
        "Unique Pedestrian Track IDs": (
            pedestrian_predictions["track_id"].nunique()
        ),
        "Ground Truth Observations": int(
            metrics["num_objects"]
        ),
        "MOTA (%)": round(
            metrics["mota"] * 100, 2
        ),
        "IDF1 (%)": round(
            metrics["idf1"] * 100, 2
        ),
        "ID Switches": int(
            metrics["num_switches"]
        ),
        "False Positives": int(
            metrics["num_false_positives"]
        ),
        "False Negatives": int(
            metrics["num_misses"]
        ),
        "FPS": round(fps, 2),
    }

    return result


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

    if not ANNOTATION_PATH.is_file():
        raise FileNotFoundError(
            f"Annotation not found: {ANNOTATION_PATH}"
        )

    create_custom_tracker()

    images = sorted(
        image
        for image in SEQUENCE_DIR.glob("*.jpg")
        if PROCESS_START <= int(image.stem) <= PROCESS_END
    )

    ground_truth = pd.read_csv(
        ANNOTATION_PATH,
        header=None,
        names=GT_COLUMNS
    )

    print("=" * 60)
    print("IMAGE DETECTIVES — BYTETRACK OPTIMIZATION")
    print("=" * 60)

    print("Sequence:", SEQUENCE_NAME)
    print("Frames processed:", len(images))
    print("Evaluation window:", EVAL_START, "-", EVAL_END)
    print("Detection confidence:", CONFIDENCE)
    print("Device:", DEVICE)

    experiments = [
        ("default_bytetrack", "bytetrack.yaml"),
        ("optimized_bytetrack", CUSTOM_TRACKER_PATH),
    ]

    results = []

    for name, tracker in experiments:

        result = run_experiment(
            images,
            name,
            tracker,
            ground_truth
        )

        results.append(result)

        pd.DataFrame(results).to_csv(
            OUTPUT_DIR / "optimization_comparison.csv",
            index=False
        )

        print(
            f"\n{name}: "
            f"{result['Pedestrian Observations']} "
            f"pedestrian observations, "
            f"MOTA {result['MOTA (%)']}%, "
            f"IDF1 {result['IDF1 (%)']}%"
        )

    print("\n" + "=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)

    print(
        pd.DataFrame(results).to_string(index=False)
    )

    print(
        "\nResults saved to:",
        OUTPUT_DIR / "optimization_comparison.csv"
    )

    print(
        "\nNote: Preliminary pedestrian-only evaluation."
    )


if __name__ == "__main__":
    main()