import io
import subprocess
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image
from ultralytics import YOLO


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Image Detectives",
    page_icon="🔍",
    layout="wide",
)


# ============================================================
# 2. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / "runs"
    / "detect"
    / "runs"
    / "train"
    / "visdrone_full"
    / "weights"
    / "best.pt"
)

TRACKING_RESULTS_DIR = BASE_DIR / "tracking_results"

EVALUATION_PATH = (
    TRACKING_RESULTS_DIR / "evaluation_results.csv"
)

TRACKING_RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 3. SESSION STATE
# ============================================================

if "video_results" not in st.session_state:
    st.session_state.video_results = None

if "image_results" not in st.session_state:
    st.session_state.image_results = None


# ============================================================
# 4. LOAD MODEL
# ============================================================

@st.cache_resource
def load_detection_model(model_path):
    return YOLO(str(model_path))


if not MODEL_PATH.is_file():
    st.error(
        f"Trained YOLOv8 model not found:\n{MODEL_PATH}"
    )
    st.stop()

model = load_detection_model(str(MODEL_PATH))


# ============================================================
# 5. HELPER FUNCTIONS
# ============================================================

def get_device():
    """Return the best available processing device."""

    if torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def convert_video_to_h264(input_path, output_path):
    """
    Convert an MP4 video into browser-compatible H.264.
    Returns conversion time.
    """

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    command = [
        ffmpeg_exe,
        "-y",
        "-i",
        str(input_path),
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(output_path),
    ]

    start = time.perf_counter()

    subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )

    return time.perf_counter() - start


def process_tracking_video(
    uploaded_file,
    confidence,
    image_size,
    device,
    trajectory_length,
    progress_bar,
    status_placeholder,
):
    """
    Perform YOLOv8 + ByteTrack tracking.

    Returns:
        Video bytes
        CSV bytes
        Tracking statistics
        Category counts
    """

    with tempfile.TemporaryDirectory() as temp_directory:

        temp_dir = Path(temp_directory)

        file_extension = (
            Path(uploaded_file.name).suffix.lower()
        )

        input_path = (
            temp_dir / f"input{file_extension}"
        )

        raw_output_path = (
            temp_dir / "tracked_raw.mp4"
        )

        h264_output_path = (
            temp_dir / "tracked_h264.mp4"
        )

        input_path.write_bytes(
            uploaded_file.getvalue()
        )

        cap = cv2.VideoCapture(str(input_path))

        if not cap.isOpened():
            raise RuntimeError(
                "Unable to open the uploaded video."
            )

        fps = cap.get(cv2.CAP_PROP_FPS)

        if fps <= 0:
            fps = 20.0

        width = int(
            cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        height = int(
            cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        total_frames = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        if width <= 0 or height <= 0:
            cap.release()
            raise RuntimeError(
                "Invalid video dimensions."
            )

        writer = cv2.VideoWriter(
            str(raw_output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

        if not writer.isOpened():
            cap.release()
            raise RuntimeError(
                "Unable to create output video."
            )

        # Separate model for each tracking session
        tracking_model = YOLO(str(MODEL_PATH))

        track_history = defaultdict(list)

        tracking_records = []

        unique_ids = set()

        class_counter = Counter()

        frame_count = 0

        detection_count = 0

        start_time = time.perf_counter()

        try:

            while True:

                success, frame = cap.read()

                if not success:
                    break

                frame_count += 1

                results = tracking_model.track(
                    source=frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    conf=confidence,
                    imgsz=image_size,
                    device=device,
                    verbose=False,
                )

                result = results[0]

                annotated_frame = result.plot()

                if result.boxes is not None:

                    boxes = result.boxes

                    detection_count += len(boxes)

                    class_ids = (
                        boxes.cls.int().cpu().tolist()
                    )

                    for class_id in class_ids:

                        class_name = (
                            tracking_model.names[class_id]
                        )

                        class_counter[class_name] += 1

                    if boxes.id is not None:

                        xyxy_boxes = (
                            boxes.xyxy.cpu().numpy()
                        )

                        track_ids = (
                            boxes.id.int().cpu().tolist()
                        )

                        confidences = (
                            boxes.conf.cpu().tolist()
                        )

                        tracked_classes = (
                            boxes.cls.int().cpu().tolist()
                        )

                        for (
                            box,
                            track_id,
                            class_id,
                            conf,
                        ) in zip(
                            xyxy_boxes,
                            track_ids,
                            tracked_classes,
                            confidences,
                        ):

                            unique_ids.add(track_id)

                            x1, y1, x2, y2 = box

                            center_x = int(
                                (x1 + x2) / 2
                            )

                            center_y = int(
                                (y1 + y2) / 2
                            )

                            # ----------------------------
                            # TRACKING TRAJECTORIES
                            # ----------------------------

                            history = track_history[track_id]

                            history.append(
                                (center_x, center_y)
                            )

                            if len(history) > trajectory_length:
                                history.pop(0)

                            if len(history) > 1:

                                points = np.array(
                                    history,
                                    dtype=np.int32,
                                ).reshape(
                                    (-1, 1, 2)
                                )

                                cv2.polylines(
                                    annotated_frame,
                                    [points],
                                    isClosed=False,
                                    color=(0, 255, 255),
                                    thickness=2,
                                )

                            # ----------------------------
                            # SAVE TRACKING DATA
                            # ----------------------------

                            tracking_records.append(
                                {
                                    "frame": frame_count,
                                    "track_id": track_id,
                                    "class_id": class_id,
                                    "class_name": (
                                        tracking_model.names[
                                            class_id
                                        ]
                                    ),
                                    "confidence": round(
                                        float(conf), 4
                                    ),
                                    "x": round(
                                        float(x1), 2
                                    ),
                                    "y": round(
                                        float(y1), 2
                                    ),
                                    "width": round(
                                        float(x2 - x1), 2
                                    ),
                                    "height": round(
                                        float(y2 - y1), 2
                                    ),
                                }
                            )

                # Save annotated frame
                writer.write(annotated_frame)

                if total_frames > 0:

                    progress_bar.progress(
                        min(
                            frame_count / total_frames,
                            1.0,
                        )
                    )

                if frame_count % 10 == 0:

                    status_placeholder.write(
                        f"Processing frame "
                        f"{frame_count}/{total_frames}"
                    )

        finally:

            cap.release()
            writer.release()

        if frame_count == 0:
            raise RuntimeError(
                "The video contained no readable frames."
            )

        elapsed_time = (
            time.perf_counter() - start_time
        )

        processing_fps = (
            frame_count / elapsed_time
            if elapsed_time > 0
            else 0
        )

        progress_bar.progress(1.0)

        # ====================================================
        # CONVERT VIDEO FOR BROWSER PLAYBACK
        # ====================================================

        conversion_time = None
        browser_compatible = False

        try:

            conversion_time = convert_video_to_h264(
                raw_output_path,
                h264_output_path,
            )

            video_bytes = (
                h264_output_path.read_bytes()
            )

            browser_compatible = True

        except (
            subprocess.CalledProcessError,
            FileNotFoundError,
            RuntimeError,
        ):

            video_bytes = (
                raw_output_path.read_bytes()
            )

        # ====================================================
        # CREATE CSV
        # ====================================================

        csv_columns = [
            "frame",
            "track_id",
            "class_id",
            "class_name",
            "confidence",
            "x",
            "y",
            "width",
            "height",
        ]

        tracking_df = pd.DataFrame(
            tracking_records,
            columns=csv_columns,
        )

        csv_bytes = tracking_df.to_csv(
            index=False
        ).encode("utf-8")

        # ====================================================
        # RETURN RESULTS
        # ====================================================

        return {
            "video_bytes": video_bytes,
            "csv_bytes": csv_bytes,
            "tracking_df": tracking_df,
            "frames": frame_count,
            "fps": processing_fps,
            "distinct_ids": len(unique_ids),
            "detections": detection_count,
            "tracked_observations": len(tracking_records),
            "class_counts": dict(class_counter),
            "conversion_time": conversion_time,
            "browser_compatible": browser_compatible,
            "input_fps": fps,
            "width": width,
            "height": height,
        }


# ============================================================
# 6. DASHBOARD HEADER
# ============================================================

st.title("🔍 Image Detectives")

st.subheader(
    "Real-Time Object Detection, Tracking, "
    "and Performance Evaluation Using YOLOv8"
)

st.write(
    "An end-to-end computer vision system using "
    "YOLOv8, ByteTrack, VisDrone, and Streamlit."
)


# ============================================================
# 7. SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Settings")

confidence = st.sidebar.slider(
    "Detection Confidence",
    min_value=0.05,
    max_value=0.95,
    value=0.30,
    step=0.05,
)

image_size = st.sidebar.selectbox(
    "Model Input Size",
    [640, 960, 1280],
    index=0,
)

default_device = get_device()

device_options = ["mps", "cpu"]

if default_device == "cpu":
    device_options = ["cpu"]

device = st.sidebar.selectbox(
    "Processing Device",
    device_options,
    index=0,
)

trajectory_length = st.sidebar.slider(
    "Tracking Trajectory Length",
    min_value=5,
    max_value=100,
    value=30,
    step=5,
)

st.sidebar.info(
    "YOLOv8 detects objects while ByteTrack "
    "maintains tracking identities across frames."
)


# ============================================================
# 8. NAVIGATION TABS
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🏠 Overview",
        "🖼️ Image Detection",
        "🎥 Video Tracking",
        "📊 Performance Evaluation",
    ]
)


# ============================================================
# TAB 1: OVERVIEW
# ============================================================

with tab1:

    st.header("Project Overview")

    st.write(
        "Image Detectives analyzes aerial images and videos "
        "to identify and track pedestrians, vehicles, "
        "bicycles, and other objects."
    )

    st.markdown(
        """
        **Technologies used**

        - YOLOv8 Nano — Object detection
        - ByteTrack — Multi-object tracking
        - VisDrone — Dataset
        - PyTorch — Deep learning
        - Streamlit — Dashboard
        - OpenCV — Video processing
        - FFmpeg — Video encoding
        """
    )

    st.subheader("Detection Model Performance")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Precision", "42.1%")

    with c2:
        st.metric("Recall", "30.3%")

    with c3:
        st.metric("mAP@50", "29.1%")

    with c4:
        st.metric("mAP@50–95", "15.9%")

    st.caption(
        "Detection metrics are from the VisDrone "
        "detection validation dataset."
    )

    st.subheader("Tracking Performance")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Preliminary MOTA", "18.4%")

    with c2:
        st.metric("Preliminary IDF1", "41.1%")

    with c3:
        st.metric("Measured Processing FPS", "36.55")

    st.warning(
        "Tracking metrics are preliminary pedestrian-only "
        "results for a selected VisDrone-MOT sequence. "
        "These are not official benchmark scores."
    )


# ============================================================
# TAB 2: IMAGE DETECTION
# ============================================================

with tab2:

    st.header("🖼️ Image Object Detection")

    uploaded_image = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"],
        key="image_upload",
    )

    if uploaded_image is not None:

        image = Image.open(
            uploaded_image
        ).convert("RGB")

        image_array = np.array(image)

        with st.spinner("Detecting objects..."):

            results = model.predict(
                source=image_array,
                conf=confidence,
                imgsz=image_size,
                device=device,
                verbose=False,
            )

        result = results[0]

        annotated_bgr = result.plot()

        annotated_rgb = cv2.cvtColor(
            annotated_bgr,
            cv2.COLOR_BGR2RGB,
        )

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("Original Image")

            st.image(
                image,
                use_container_width=True,
            )

        with col2:

            st.subheader("Detected Objects")

            st.image(
                annotated_rgb,
                use_container_width=True,
            )

        st.subheader("Detection Summary")

        class_ids = []

        if result.boxes is not None:

            class_ids = (
                result.boxes.cls.int().cpu().tolist()
            )

        class_names = [
            model.names[int(class_id)]
            for class_id in class_ids
        ]

        counts = Counter(class_names)

        st.metric(
            "Total Detected Objects",
            sum(counts.values()),
        )

        if counts:

            count_df = pd.DataFrame(
                [
                    {
                        "Object": name,
                        "Count": count,
                    }
                    for name, count in counts.items()
                ]
            )

            st.bar_chart(
                count_df.set_index("Object")
            )

            st.dataframe(
                count_df,
                hide_index=True,
                use_container_width=True,
            )

        else:

            st.warning(
                "No objects detected. Try reducing "
                "the confidence threshold."
            )

        success, encoded_image = cv2.imencode(
            ".jpg",
            annotated_bgr,
        )

        if success:

            st.download_button(
                "⬇️ Download Annotated Image",
                data=encoded_image.tobytes(),
                file_name="image_detectives_detection.jpg",
                mime="image/jpeg",
            )

    else:

        st.info(
            "Upload an image to begin object detection."
        )


# ============================================================
# TAB 3: VIDEO TRACKING
# ============================================================

with tab3:

    st.header("🎥 Video Object Tracking")

    st.write(
        "Upload a video to detect and track moving objects "
        "using YOLOv8 and ByteTrack."
    )

    uploaded_video = st.file_uploader(
        "Upload Video",
        type=["mp4", "mov", "avi"],
        key="video_upload",
    )

    if uploaded_video is not None:

        st.subheader("Original Video")

        st.video(uploaded_video)

        if st.button(
            "🚀 Start Video Tracking",
            key="start_tracking",
        ):

            progress_bar = st.progress(0)

            status_placeholder = st.empty()

            with st.spinner(
                "Processing video and tracking objects..."
            ):

                try:

                    results = process_tracking_video(
                        uploaded_file=uploaded_video,
                        confidence=confidence,
                        image_size=image_size,
                        device=device,
                        trajectory_length=trajectory_length,
                        progress_bar=progress_bar,
                        status_placeholder=status_placeholder,
                    )

                    st.session_state.video_results = results

                    status_placeholder.success(
                        "Video tracking completed!"
                    )

                except Exception as error:

                    st.session_state.video_results = None

                    st.error(
                        f"Video processing failed: {error}"
                    )

    else:

        st.info(
            "Upload a video to begin tracking."
        )

    # ========================================================
    # DISPLAY SAVED RESULTS
    # ========================================================

    video_results = st.session_state.video_results

    if video_results is not None:

        st.divider()

        st.header("📊 Video Tracking Results")

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric(
                "Frames Processed",
                video_results["frames"],
            )

        with c2:
            st.metric(
                "Distinct Track IDs",
                video_results["distinct_ids"],
            )

        with c3:
            st.metric(
                "Tracked Observations",
                video_results["tracked_observations"],
            )

        with c4:
            st.metric(
                "Processing FPS",
                f'{video_results["fps"]:.2f}',
            )

        st.caption(
            "Distinct track IDs are tracker-generated identities. "
            "They do not necessarily equal the number of unique "
            "real-world objects."
        )

        st.caption(
            "Processing FPS includes decoding, detection, "
            "tracking, trajectory drawing, and raw video writing. "
            "H.264 conversion time is excluded."
        )

        # ====================================================
        # CLASS-WISE DETECTION COUNTS
        # ====================================================

        class_counts = video_results["class_counts"]

        if class_counts:

            st.subheader("Object Counts by Category")

            class_df = pd.DataFrame(
                [
                    {
                        "Object": name,
                        "Detections": count,
                    }
                    for name, count in class_counts.items()
                ]
            )

            st.bar_chart(
                class_df.set_index("Object")
            )

            st.dataframe(
                class_df,
                hide_index=True,
                use_container_width=True,
            )

            st.caption(
                "Counts represent detections across all frames, "
                "not distinct physical objects."
            )

        # ====================================================
        # TRACKING CSV
        # ====================================================

        st.subheader("📋 Tracking Data")

        tracking_df = video_results["tracking_df"]

        if not tracking_df.empty:

            st.dataframe(
                tracking_df.head(100),
                hide_index=True,
                use_container_width=True,
            )

            st.caption(
                "Showing the first 100 records. "
                "The CSV contains all tracking observations."
            )

        else:

            st.warning(
                "No tracked objects were recorded."
            )

        st.download_button(
            "⬇️ Download Tracking CSV",
            data=video_results["csv_bytes"],
            file_name="image_detectives_tracking.csv",
            mime="text/csv",
            key="download_tracking_csv",
        )

        # ====================================================
        # PROCESSED VIDEO
        # ====================================================

        st.subheader("🎬 Processed Video")

        if video_results["browser_compatible"]:

            st.video(
                video_results["video_bytes"],
                format="video/mp4",
            )

            st.success(
                "Processed video is ready for playback."
            )

        else:

            st.warning(
                "H.264 conversion was unavailable. "
                "Download the MP4V video for local playback."
            )

        conversion_time = (
            video_results["conversion_time"]
        )

        if conversion_time is not None:

            st.caption(
                f"H.264 conversion time: "
                f"{conversion_time:.2f} seconds"
            )

        video_filename = (
            "image_detectives_tracked.mp4"
            if video_results["browser_compatible"]
            else "image_detectives_tracked_mp4v.mp4"
        )

        st.download_button(
            "⬇️ Download Tracked Video",
            data=video_results["video_bytes"],
            file_name=video_filename,
            mime="video/mp4",
            key="download_tracked_video",
        )


# ============================================================
# TAB 4: PERFORMANCE EVALUATION
# ============================================================

with tab4:

    st.header("📊 Performance Evaluation")

    # ========================================================
    # DETECTION METRICS
    # ========================================================

    st.subheader("YOLOv8 Detection Metrics")

    detection_metrics = pd.DataFrame(
        {
            "Metric": [
                "Precision",
                "Recall",
                "mAP@50",
                "mAP@50–95",
            ],
            "Score (%)": [
                42.1,
                30.3,
                29.1,
                15.9,
            ],
        }
    )

    st.dataframe(
        detection_metrics,
        hide_index=True,
        use_container_width=True,
    )

    st.bar_chart(
        detection_metrics.set_index("Metric")
    )

    # ========================================================
    # LATEST TRACKING EVALUATION
    # ========================================================

    st.subheader("ByteTrack Evaluation Results")

    if EVALUATION_PATH.exists():

        evaluation_df = pd.read_csv(
            EVALUATION_PATH
        )

        st.dataframe(
            evaluation_df,
            hide_index=True,
            use_container_width=True,
        )

        st.download_button(
            "⬇️ Download Evaluation CSV",
            data=EVALUATION_PATH.read_bytes(),
            file_name="tracking_evaluation.csv",
            mime="text/csv",
        )

    else:

        st.warning(
            "Tracking evaluation results not found."
        )

    # ========================================================
    # EXPERIMENT COMPARISON
    # ========================================================

    st.subheader("Confidence Threshold Experiments")

    experiment_df = pd.DataFrame(
        {
            "Confidence": [
                0.10,
                0.20,
                0.25,
                0.30,
            ],
            "MOTA (%)": [
                2.3,
                10.1,
                12.9,
                18.4,
            ],
            "IDF1 (%)": [
                37.1,
                38.3,
                39.0,
                41.1,
            ],
            "FPS": [
                27.64,
                32.65,
                34.30,
                36.55,
            ],
            "ID Switches": [
                127,
                129,
                136,
                123,
            ],
        }
    )

    st.dataframe(
        experiment_df,
        hide_index=True,
        use_container_width=True,
    )

    st.subheader("Tracking Accuracy Comparison")

    st.line_chart(
        experiment_df.set_index("Confidence")[
            ["MOTA (%)", "IDF1 (%)"]
        ]
    )

    st.subheader("Processing Speed Comparison")

    st.line_chart(
        experiment_df.set_index("Confidence")[
            ["FPS"]
        ]
    )

    st.caption(
        "Confidence labels are based on the planned "
        "experimental sequence. Tracking metrics are "
        "preliminary pedestrian-only results and are "
        "not official VisDrone benchmark scores."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Image Detectives | YOLOv8 | ByteTrack | "
    "VisDrone | Streamlit"
)