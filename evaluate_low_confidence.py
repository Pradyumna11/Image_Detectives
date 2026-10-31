from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

# NumPy 2.x compatibility
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

SEQUENCE_DIR = DATASET_DIR / "sequences" / SEQUENCE_NAME

ANNOTATION_PATH = (
    DATASET_DIR
    / "annotations"
    / f"{SEQUENCE_NAME}.txt"
)

OUTPUT_DIR = (
    BASE_DIR
    / "tracking_results/low_confidence"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CONFIDENCE_LEVELS = [0.05, 0.10, 0.20, 0.30]

IMAGE_SIZE = 640

DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# Process earlier frames to maintain tracking state.
PROCESS_START = 1
PROCESS_END = 600

# Only calculate metrics for these frames.
EVAL_START = 301
EVAL_END = 600

IOU_THRESHOLD = 0.50

YOLO_PEDESTRIAN_CLASS = 0
VISDRONE_PEDESTRIAN_CLASS = 1

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
# IGNORED REGION HANDLING
# ============================================================

def overlaps_ignored_region(box, ignored_boxes):

    x, y, w, h = box

    area = w * h

    if area <= 0:
        return True

    for rx, ry, rw, rh in ignored_boxes:

        ix1 = max(x, rx)
        iy1 = max(y, ry)

        ix2 = min(x + w, rx + rw)
        iy2 = min(y + h, ry + rh)

        intersection = (
            max(0, ix2 - ix1)
            * max(0, iy2 - iy1)
        )

        if intersection / area > 0.5:
            return True

    return False


# ============================================================
# TRACKING EVALUATION
# ============================================================

def evaluate_tracking(gt, pred, frame_numbers):

    gt = gt[
        gt["frame"].isin(frame_numbers)
    ].copy()

    pred = pred[
        pred["frame"].isin(frame_numbers)
    ].copy()

    gt_pedestrians = gt[
        (gt["class_id"] == VISDRONE_PEDESTRIAN_CLASS)
        & (gt["score"] == 1)
        & (gt["track_id"] > 0)
    ].copy()

    predicted_pedestrians = pred[
        pred["class_id"] == YOLO_PEDESTRIAN_CLASS
    ].copy()

    ignored = gt[
        (gt["class_id"] == 0)
        | (gt["score"] == 0)
    ].copy()

    gt_groups = {
        int(frame): group
        for frame, group in gt_pedestrians.groupby("frame")
    }

    pred_groups = {
        int(frame): group
        for frame, group in predicted_pedestrians.groupby("frame")
    }

    ignored_groups = {
        int(frame): group[
            ["x", "y", "width", "height"]
        ].to_numpy(dtype=float)
        for frame, group in ignored.groupby("frame")
    }

    accumulator = mm.MOTAccumulator(auto_id=True)

    empty_gt = gt_pedestrians.iloc[0:0]
    empty_pred = predicted_pedestrians.iloc[0:0]

    for frame in frame_numbers:

        frame_gt = gt_groups.get(frame, empty_gt)
        frame_pred = pred_groups.get(frame, empty_pred)

        ignored_boxes = ignored_groups.get(
            frame,
            np.empty((0, 4), dtype=float),
        )

        if len(ignored_boxes) and not frame_pred.empty:

            keep = []

            for row in frame_pred.itertuples():

                box = (
                    row.x,
                    row.y,
                    row.width,
                    row.height,
                )

                keep.append(
                    not overlaps_ignored_region(
                        box,
                        ignored_boxes,
                    )
                )

            frame_pred = frame_pred.loc[keep]

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
            max_iou=1.0 - IOU_THRESHOLD,
        )

        accumulator.update(
            gt_ids,
            pred_ids,
            distances,
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
        name="pedestrian_tracking",
    )

    return summary.iloc[0].to_dict()


# ============================================================
# RUN EXPERIMENT
# ============================================================

def run_experiment(images, confidence, ground_truth):

    print(f"\nTesting confidence: {confidence:.2f}")

    model = YOLO(str(MODEL_PATH))

    predictions = []

    raw_pedestrian_detections = 0
    tracked_pedestrian_observations = 0

    frames_with_pedestrian_tracks = 0

    frame_numbers = list(range(EVAL_START, EVAL_END + 1))

    start_time = time.perf_counter()

    for index, image_path in enumerate(images, start=1):

        frame_number = int(image_path.stem)

        results = model.track(
            source=str(image_path),
            tracker="bytetrack.yaml",
            persist=True,
            conf=confidence,
            imgsz=IMAGE_SIZE,
            device=DEVICE,
            verbose=False,
        )

        boxes = results[0].boxes

        if boxes is not None:

            classes = boxes.cls.int().cpu().tolist()

            if EVAL_START <= frame_number <= EVAL_END:

                raw_pedestrian_detections += classes.count(
                    YOLO_PEDESTRIAN_CLASS
                )

            if boxes.id is not None:

                coords = boxes.xyxy.cpu().numpy()
                ids = boxes.id.int().cpu().tolist()
                confidences = boxes.conf.cpu().tolist()

                pedestrian_tracks_this_frame = 0

                for box, track_id, class_id, score in zip(
                    coords,
                    ids,
                    classes,
                    confidences,
                ):

                    x1, y1, x2, y2 = box

                    if EVAL_START <= frame_number <= EVAL_END:

                        predictions.append(
                            [
                                frame_number,
                                track_id,
                                float(x1),
                                float(y1),
                                float(x2 - x1),
                                float(y2 - y1),
                                float(score),
                                class_id,
                            ]
                        )

                        if class_id == YOLO_PEDESTRIAN_CLASS:

                            pedestrian_tracks_this_frame += 1

                            tracked_pedestrian_observations += 1

                if pedestrian_tracks_this_frame > 0:
                    frames_with_pedestrian_tracks += 1

        if index % 100 == 0 or index == len(images):

            print(
                f"  Processed {index}/{len(images)} frames"
            )

    elapsed = time.perf_counter() - start_time

    fps = len(images) / elapsed if elapsed > 0 else 0

    predictions_df = pd.DataFrame(
        predictions,
        columns=PRED_COLUMNS,
    )

    predictions_df.to_csv(
        OUTPUT_DIR / f"predictions_conf_{confidence:.2f}.csv",
        index=False,
    )

    metrics = evaluate_tracking(
        ground_truth,
        predictions_df,
        frame_numbers,
    )

    pedestrian_predictions = predictions_df[
        predictions_df["class_id"] == YOLO_PEDESTRIAN_CLASS
    ]

    return {
        "Confidence": confidence,
        "Processed Frames": len(images),
        "Evaluated Frames": len(frame_numbers),
        "Raw Pedestrian Detections": raw_pedestrian_detections,
        "Pedestrian Tracked Observations": tracked_pedestrian_observations,
        "Frames With Pedestrian Tracks": frames_with_pedestrian_tracks,
        "Distinct Pedestrian Track IDs": (
            pedestrian_predictions["track_id"].nunique()
        ),
        "MOTA (%)": round(metrics["mota"] * 100, 2),
        "IDF1 (%)": round(metrics["idf1"] * 100, 2),
        "ID Switches": int(metrics["num_switches"]),
        "False Positives": int(metrics["num_false_positives"]),
        "False Negatives": int(metrics["num_misses"]),
        "Ground Truth Observations": int(metrics["num_objects"]),
        "FPS": round(fps, 2),
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

    if not ANNOTATION_PATH.is_file():
        raise FileNotFoundError(
            f"Annotations not found: {ANNOTATION_PATH}"
        )

    images = sorted(
        SEQUENCE_DIR.glob("*.jpg")
    )

    images = [
        image
        for image in images
        if PROCESS_START <= int(image.stem) <= PROCESS_END
    ]

    ground_truth = pd.read_csv(
        ANNOTATION_PATH,
        header=None,
        names=GT_COLUMNS,
    )

    print("=" * 65)
    print("IMAGE DETECTIVES — CORRECTED TRACKING EXPERIMENT")
    print("=" * 65)

    print("Sequence:", SEQUENCE_NAME)
    print("Processing frames:", PROCESS_START, "-", PROCESS_END)
    print("Evaluation frames:", EVAL_START, "-", EVAL_END)
    print("Device:", DEVICE)

    experiment_results = []

    for confidence in CONFIDENCE_LEVELS:

        result = run_experiment(
            images,
            confidence,
            ground_truth,
        )

        experiment_results.append(result)

        pd.DataFrame(experiment_results).to_csv(
            OUTPUT_DIR / "confidence_comparison_corrected.csv",
            index=False,
        )

        print(
            f"  Confidence: {confidence:.2f} | "
            f"Tracked pedestrians: "
            f"{result['Pedestrian Tracked Observations']} | "
            f"MOTA: {result['MOTA (%)']}% | "
            f"IDF1: {result['IDF1 (%)']}%"
        )

    results_df = pd.DataFrame(experiment_results)

    print("\n" + "=" * 65)
    print("FINAL CONFIDENCE COMPARISON")
    print("=" * 65)

    print(results_df.to_string(index=False))

    print(
        "\nSaved to:",
        OUTPUT_DIR / "confidence_comparison_corrected.csv",
    )

    print(
        "\nNote: Preliminary pedestrian-only evaluation; "
        "not official VisDrone benchmark scoring."
    )


if __name__ == "__main__":
    main()