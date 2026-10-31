from pathlib import Path
from collections import defaultdict
import time

import numpy as np

# Compatibility fix for motmetrics with NumPy 2.x
if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=float: np.asarray(a, dtype=dtype)

import pandas as pd
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

DATASET_DIR = (
    BASE_DIR
    / "videos/visdrone_mot/VisDrone2019-MOT-val"
)

SEQUENCES_DIR = DATASET_DIR / "sequences"
ANNOTATIONS_DIR = DATASET_DIR / "annotations"

OUTPUT_DIR = BASE_DIR / "tracking_results" / "multi_sequence"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CONFIDENCE = 0.30
IMAGE_SIZE = 640
DEVICE = "mps"

# Set to None to evaluate every frame in every sequence.
# Use 300 to reproduce the frame limit from your earlier test.
MAX_FRAMES = None

IOU_THRESHOLD = 0.50

# Evaluate pedestrians first.
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
# HELPERS
# ============================================================

def overlap_with_ignored_region(box, ignored_boxes):
    """
    Return True when more than 50% of a predicted box
    overlaps an ignored region.

    This is a simplified experimental rule, not the
    official VisDrone ignored-region protocol.
    """

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


def evaluate_predictions(gt, pred, frame_count):
    """
    Preliminary pedestrian-only tracking evaluation.

    Uses IoU >= 0.5 for matching and simplified handling
    of ignored regions.
    """

    gt = gt[
        gt["frame"].between(1, frame_count)
    ].copy()

    pred = pred[
        pred["frame"].between(1, frame_count)
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

    gt_frames = {
        int(frame): group
        for frame, group in gt_pedestrians.groupby("frame")
    }

    pred_frames = {
        int(frame): group
        for frame, group in predicted_pedestrians.groupby("frame")
    }

    ignored_frames = {
        int(frame): group[
            ["x", "y", "width", "height"]
        ].to_numpy(dtype=float)
        for frame, group in ignored.groupby("frame")
    }

    acc = mm.MOTAccumulator(auto_id=True)

    empty_gt = gt_pedestrians.iloc[0:0]
    empty_pred = predicted_pedestrians.iloc[0:0]

    for frame_number in range(1, frame_count + 1):

        frame_gt = gt_frames.get(
            frame_number,
            empty_gt,
        )

        frame_pred = pred_frames.get(
            frame_number,
            empty_pred,
        )

        ignored_boxes = ignored_frames.get(
            frame_number,
            np.empty((0, 4), dtype=float),
        )

        if len(ignored_boxes) and not frame_pred.empty:
            keep = [
                not overlap_with_ignored_region(
                    (
                        row.x,
                        row.y,
                        row.width,
                        row.height,
                    ),
                    ignored_boxes,
                )
                for row in frame_pred.itertuples()
            ]

            frame_pred = frame_pred.loc[keep]

        gt_ids = (
            frame_gt["track_id"]
            .astype(int)
            .tolist()
        )

        predicted_ids = (
            frame_pred["track_id"]
            .astype(int)
            .tolist()
        )

        gt_boxes = frame_gt[
            ["x", "y", "width", "height"]
        ].to_numpy(dtype=float).reshape(-1, 4)

        predicted_boxes = frame_pred[
            ["x", "y", "width", "height"]
        ].to_numpy(dtype=float).reshape(-1, 4)

        distances = mm.distances.iou_matrix(
            gt_boxes,
            predicted_boxes,
            max_iou=1.0 - IOU_THRESHOLD,
        )

        acc.update(
            gt_ids,
            predicted_ids,
            distances,
        )

    metrics = [
        "num_frames",
        "mota",
        "idf1",
        "num_switches",
        "num_false_positives",
        "num_misses",
        "num_objects",
        "idtp",
        "idfp",
        "idfn",
    ]

    mh = mm.metrics.create()

    summary = mh.compute(
        acc,
        metrics=metrics,
        name="tracking",
    )

    return summary.iloc[0].to_dict()


# ============================================================
# TRACK ONE SEQUENCE
# ============================================================

def process_sequence(sequence_dir, annotation_path):

    sequence_name = sequence_dir.name

    images = sorted(
        sequence_dir.glob("*.jpg")
    )

    if MAX_FRAMES is not None:
        images = images[:MAX_FRAMES]

    if not images:
        raise RuntimeError(
            f"No JPG frames found in {sequence_name}"
        )

    gt = pd.read_csv(
        annotation_path,
        header=None,
        names=GT_COLUMNS,
    )

    # Fresh model ensures tracker state is not carried
    # from one sequence into another.
    model = YOLO(str(MODEL_PATH))

    predictions = []

    start = time.perf_counter()

    for index, image_path in enumerate(images, start=1):

        results = model.track(
            source=str(image_path),
            tracker="bytetrack.yaml",
            persist=True,
            conf=CONFIDENCE,
            imgsz=IMAGE_SIZE,
            device=DEVICE,
            verbose=False,
        )

        result = results[0]

        if (
            result.boxes is not None
            and result.boxes.id is not None
        ):

            boxes = result.boxes

            coords = boxes.xyxy.cpu().numpy()
            ids = boxes.id.int().cpu().tolist()
            classes = boxes.cls.int().cpu().tolist()
            confidences = boxes.conf.cpu().tolist()

            for box, track_id, class_id, confidence in zip(
                coords,
                ids,
                classes,
                confidences,
            ):

                x1, y1, x2, y2 = box

                predictions.append(
                    [
                        index,
                        track_id,
                        float(x1),
                        float(y1),
                        float(x2 - x1),
                        float(y2 - y1),
                        float(confidence),
                        class_id,
                    ]
                )

        if index % 100 == 0 or index == len(images):
            print(
                f"  Frames processed: "
                f"{index}/{len(images)}"
            )

    elapsed = time.perf_counter() - start

    processing_fps = (
        len(images) / elapsed
        if elapsed > 0
        else 0
    )

    pred = pd.DataFrame(
        predictions,
        columns=PRED_COLUMNS,
    )

    prediction_path = (
        OUTPUT_DIR / f"{sequence_name}_predictions.csv"
    )

    pred.to_csv(
        prediction_path,
        index=False,
    )

    metrics = evaluate_predictions(
        gt,
        pred,
        len(images),
    )

    return {
        "Sequence": sequence_name,
        "Frames": len(images),
        "Tracked Observations": len(pred),
        "MOTA (%)": round(metrics["mota"] * 100, 2),
        "IDF1 (%)": round(metrics["idf1"] * 100, 2),
        "ID Switches": int(metrics["num_switches"]),
        "False Positives": int(metrics["num_false_positives"]),
        "False Negatives": int(metrics["num_misses"]),
        "GT Observations": int(metrics["num_objects"]),
        "FPS": round(processing_fps, 2),
        "Confidence": CONFIDENCE,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    if not SEQUENCES_DIR.is_dir():
        raise FileNotFoundError(
            f"Sequences directory not found: {SEQUENCES_DIR}"
        )

    sequences = sorted(
        folder
        for folder in SEQUENCES_DIR.iterdir()
        if folder.is_dir()
    )

    print("=" * 65)
    print("IMAGE DETECTIVES — MULTI-SEQUENCE EVALUATION")
    print("=" * 65)

    print("Sequences found:", len(sequences))
    print("Confidence:", CONFIDENCE)
    print("Image size:", IMAGE_SIZE)
    print("Device:", DEVICE)
    print("Maximum frames per sequence:", MAX_FRAMES)
    print()

    all_results = []

    for number, sequence_dir in enumerate(
        sequences,
        start=1,
    ):

        sequence_name = sequence_dir.name

        annotation_path = (
            ANNOTATIONS_DIR / f"{sequence_name}.txt"
        )

        print(
            f"\n[{number}/{len(sequences)}] "
            f"Processing {sequence_name}"
        )

        if not annotation_path.is_file():
            print(
                "  Skipping: annotation file not found."
            )
            continue

        try:

            result = process_sequence(
                sequence_dir,
                annotation_path,
            )

            all_results.append(result)

            print(
                f"  MOTA: {result['MOTA (%)']}%"
            )

            print(
                f"  IDF1: {result['IDF1 (%)']}%"
            )

            print(
                f"  ID switches: "
                f"{result['ID Switches']}"
            )

            print(
                f"  Processing FPS: "
                f"{result['FPS']}"
            )

            # Save after each successful sequence
            pd.DataFrame(all_results).to_csv(
                OUTPUT_DIR / "all_sequences_results.csv",
                index=False,
            )

        except Exception as error:
            print(
                f"  Error: {error}"
            )

    if not all_results:
        print("\nNo sequences were evaluated.")
        return

    results_df = pd.DataFrame(all_results)

    print("\n" + "=" * 65)
    print("FINAL RESULTS")
    print("=" * 65)

    print(
        results_df[
            [
                "Sequence",
                "Frames",
                "MOTA (%)",
                "IDF1 (%)",
                "ID Switches",
                "FPS",
            ]
        ].to_string(index=False)
    )

    print(
        "\nResults saved to:",
        OUTPUT_DIR / "all_sequences_results.csv",
    )

    print(
        "\nNote: Results use a simplified pedestrian-only "
        "evaluation protocol, not official VisDrone scoring."
    )


if __name__ == "__main__":
    main()