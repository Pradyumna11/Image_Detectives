import pandas as pd
import numpy as np

# Compatibility fix for motmetrics with NumPy 2.x
if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=float: np.asarray(a, dtype=dtype)

import motmetrics as mm
# File paths
GT_PATH = (
    "videos/visdrone_mot/"
    "VisDrone2019-MOT-val/"
    "annotations/uav0000086_00000_v.txt"
)

PRED_PATH = "tracking_results/tracking_predictions.csv"

# Load ground truth
gt = pd.read_csv(
    GT_PATH,
    header=None,
    names=[
        "frame", "track_id", "x", "y",
        "width", "height", "score",
        "class_id", "truncation", "occlusion"
    ]
)

# Load predicted tracking results
pred = pd.read_csv(PRED_PATH)

# Evaluate only frames 1–300
gt = gt[gt["frame"].between(1, 300)]
pred = pred[pred["frame"].between(1, 300)]

# Keep valid ground-truth pedestrians
gt_ped = gt[
    (gt["class_id"] == 1) &
    (gt["score"] == 1) &
    (gt["track_id"] > 0)
].copy()

# Keep YOLO pedestrian predictions
pred_ped = pred[
    pred["class_id"] == 0
].copy()

# Exclude predictions substantially inside ignored regions
ignored = gt[
    (gt["class_id"] == 0) |
    (gt["score"] == 0)
]

def ignored_prediction(box, regions):
    x, y, w, h = box
    area = w * h

    if area <= 0:
        return True

    for _, region in regions.iterrows():
        rx = region["x"]
        ry = region["y"]
        rw = region["width"]
        rh = region["height"]

        ix1 = max(x, rx)
        iy1 = max(y, ry)
        ix2 = min(x + w, rx + rw)
        iy2 = min(y + h, ry + rh)

        intersection = (
            max(0, ix2 - ix1) *
            max(0, iy2 - iy1)
        )

        if intersection / area > 0.5:
            return True

    return False


# Create MOT accumulator
acc = mm.MOTAccumulator(auto_id=True)

for frame in range(1, 301):

    frame_gt = gt_ped[gt_ped["frame"] == frame]

    frame_pred = pred_ped[pred_ped["frame"] == frame].copy()

    frame_ignored = ignored[ignored["frame"] == frame]

    # Remove detections inside ignored regions
    if not frame_ignored.empty and not frame_pred.empty:

        keep = []

        for _, row in frame_pred.iterrows():

            box = [
                row["x"], row["y"],
                row["width"], row["height"]
            ]

            keep.append(
                not ignored_prediction(box, frame_ignored)
            )

        frame_pred = frame_pred.loc[keep]

    gt_ids = frame_gt["track_id"].astype(int).tolist()

    pred_ids = frame_pred["track_id"].astype(int).tolist()

    gt_boxes = frame_gt[
        ["x", "y", "width", "height"]
    ].to_numpy(dtype=float).reshape(-1, 4)

    pred_boxes = frame_pred[
        ["x", "y", "width", "height"]
    ].to_numpy(dtype=float).reshape(-1, 4)

    # IoU threshold = 0.5
    distances = mm.distances.iou_matrix(
        gt_boxes,
        pred_boxes,
        max_iou=0.5
    )

    acc.update(
        gt_ids,
        pred_ids,
        distances
    )

# Calculate metrics
mh = mm.metrics.create()

summary = mh.compute(
    acc,
    metrics=[
        "num_frames",
        "mota",
        "idf1",
        "num_switches",
        "num_false_positives",
        "num_misses",
        "num_objects"
    ],
    name="YOLOv8_ByteTrack"
)

print("\n===== TRACKING EVALUATION =====")

print(
    mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
)

# Save results
output = "tracking_results/evaluation_results.csv"

summary.to_csv(output)

print("\nEvaluation completed!")
print("Saved results to:", output)