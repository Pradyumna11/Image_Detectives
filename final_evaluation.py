from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

TRACKING_DIR = BASE_DIR / "tracking_results"

OUTPUT_DIR = BASE_DIR / "final_results"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# INPUT FILES
# ============================================================

MULTI_SEQUENCE_FILE = (
    TRACKING_DIR
    / "multi_sequence"
    / "baseline_conf030.csv"
)

CONFIDENCE_FILE = (
    TRACKING_DIR
    / "pedestrian_analysis"
    / "confidence_analysis.csv"
)

RESOLUTION_FILE = (
    TRACKING_DIR
    / "resolution_experiments"
    / "resolution_comparison.csv"
)

BYTETRACK_FILE = (
    TRACKING_DIR
    / "bytetrack_debug"
    / "bytetrack_debug_results.csv"
)


# ============================================================
# DETECTION PERFORMANCE
# ============================================================

DETECTION_METRICS = {
    "Model": "YOLOv8n",
    "Epochs": 30,
    "Precision (%)": 42.1,
    "Recall (%)": 30.3,
    "mAP50 (%)": 29.1,
    "mAP50-95 (%)": 15.9,
}


# ============================================================
# LOAD CSV
# ============================================================

def load_csv(path, name):

    if not path.is_file():

        print(f"WARNING: Missing {name}: {path}")
        return None

    df = pd.read_csv(path)

    print(
        f"Loaded {name}: "
        f"{len(df)} rows"
    )

    return df


# ============================================================
# SAVE DETECTION SUMMARY
# ============================================================

def save_detection_summary():

    df = pd.DataFrame(
        [DETECTION_METRICS]
    )

    output = (
        OUTPUT_DIR
        / "detection_summary.csv"
    )

    df.to_csv(
        output,
        index=False
    )

    print("\nDETECTION PERFORMANCE")
    print(df.to_string(index=False))

    return df


# ============================================================
# TRACKING SUMMARY
# ============================================================

def process_tracking_summary(df):

    if df is None:
        return

    print("\nTRACKING PERFORMANCE")
    print(df.to_string(index=False))

    output = (
        OUTPUT_DIR
        / "tracking_summary.csv"
    )

    df.to_csv(
        output,
        index=False
    )

    # Identify possible metric columns.
    numeric_columns = df.select_dtypes(
        include="number"
    ).columns

    print("\nTRACKING COLUMN CHECK")

    for column in numeric_columns:

        values = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        invalid_count = (
            ~np.isfinite(values.to_numpy(dtype=float))
        ).sum()

        if invalid_count:

            print(
                f"WARNING: {column} contains "
                f"{invalid_count} non-finite values"
            )

    # Do not average percentages across sequences.
    # A meaningful overall MOTA requires aggregate
    # false positives, misses, ID switches,
    # and ground-truth observation counts.
    print(
        "\nNOTE: Sequence-level MOTA and IDF1 "
        "have not been averaged."
    )

    print(
        "Overall tracking metrics require "
        "aggregation from event/count data."
    )


# ============================================================
# EXPERIMENT SUMMARIES
# ============================================================

def save_experiment(
    df,
    name,
    output_filename
):

    if df is None:
        return

    output = (
        OUTPUT_DIR
        / output_filename
    )

    df.to_csv(
        output,
        index=False
    )

    print(f"\n{name.upper()}")
    print(df.to_string(index=False))


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print("IMAGE DETECTIVES — FINAL EVALUATION")
    print("=" * 65)

    save_detection_summary()

    tracking_df = load_csv(
        MULTI_SEQUENCE_FILE,
        "Multi-sequence evaluation"
    )

    process_tracking_summary(
        tracking_df
    )

    confidence_df = load_csv(
        CONFIDENCE_FILE,
        "Confidence analysis"
    )

    save_experiment(
        confidence_df,
        "Confidence experiment",
        "confidence_summary.csv"
    )

    resolution_df = load_csv(
        RESOLUTION_FILE,
        "Resolution experiment"
    )

    save_experiment(
        resolution_df,
        "Resolution experiment",
        "resolution_summary.csv"
    )

    tracker_df = load_csv(
        BYTETRACK_FILE,
        "ByteTrack debugging"
    )

    save_experiment(
        tracker_df,
        "ByteTrack configuration comparison",
        "bytetrack_summary.csv"
    )

    print("\n" + "=" * 65)
    print("EVALUATION SUMMARY COMPLETE")
    print("=" * 65)

    print("\nResults directory:")
    print(OUTPUT_DIR)

    print(
        "\nIMPORTANT: These tracking metrics "
        "use a simplified pedestrian-only "
        "evaluation protocol."
    )


if __name__ == "__main__":
    main()