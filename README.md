# Image Detectives
## Real-Time Object Detection, Tracking, and Performance Evaluation Using YOLOv8

An end-to-end computer vision project that combines YOLOv8 object detection, ByteTrack multi-object tracking, and an interactive Streamlit dashboard for analyzing aerial images and videos.

## Project Overview

Image Detectives is designed to detect, classify, and track multiple objects in aerial imagery using deep learning.

The system uses a custom-trained YOLOv8n model and the VisDrone dataset to identify objects such as pedestrians, cars, buses, trucks, bicycles, and motorcycles.

ByteTrack assigns tracking IDs to detected objects across consecutive video frames.

The project also evaluates detection accuracy, tracking consistency, and processing performance.

## Key Features

- Object detection using YOLOv8
- Multi-object tracking using ByteTrack
- Custom model training using VisDrone
- Object identification with bounding boxes
- Confidence scores and object counts
- Persistent tracking IDs
- Object trajectory visualization
- Image and video upload support
- Interactive Streamlit dashboard
- Downloadable annotated videos and tracking CSV files
- Detection and tracking performance evaluation

## Technologies Used

| Category | Technologies |
|---|---|
| Programming | Python |
| Deep Learning | PyTorch |
| Object Detection | YOLOv8n, Ultralytics |
| Object Tracking | ByteTrack |
| Dataset | VisDrone |
| Computer Vision | OpenCV |
| Dashboard | Streamlit |
| Data Processing | Pandas, NumPy |
| Evaluation | motmetrics |
| Visualization | Matplotlib, Streamlit |
| Video Processing | FFmpeg |

## System Architecture

The project follows this workflow:

```text
Input Image / Video
        |
        v
Image Preprocessing
        |
        v
Custom YOLOv8 Model
        |
        v
Object Detection
        |
        +-------------------+
        |                   |
        v                   v
Image Annotation       ByteTrack
                            |
                            v
                    Object Tracking IDs
                            |
                            v
                    Trajectory Visualization
                            |
                            v
                    Streamlit Dashboard
                            |
                            v
                  Results and CSV Export
```

## Dataset

The project uses the VisDrone dataset, which contains aerial images and video sequences collected using drones.

The dataset includes object categories such as:

1. Pedestrian
2. People
3. Bicycle
4. Car
5. Van
6. Truck
7. Tricycle
8. Awning-tricycle
9. Bus
10. Motor

Dataset source:

https://github.com/VisDrone/VisDrone-Dataset

The dataset is not included in this repository.

## Model Training

The object detection model was trained using YOLOv8n.

| Parameter | Value |
|---|---|
| Architecture | YOLOv8n |
| Training epochs | 30 |
| Dataset | VisDrone |
| Framework | Ultralytics |

The trained model weights are excluded from this repository.

## Object Detection Performance

| Metric | Result |
|---|---:|
| Precision | 42.1% |
| Recall | 30.3% |
| mAP@50 | 29.1% |
| mAP@50–95 | 15.9% |

These values were obtained from the project's YOLOv8 validation experiment.

## Multi-Object Tracking Evaluation

ByteTrack was evaluated on seven VisDrone video sequences.

| Sequence | Frames | MOTA | IDF1 |
|---|---:|---:|---:|
| uav0000086_00000_v | 464 | 21.78% | 40.67% |
| uav0000117_02622_v | 349 | 15.91% | 20.49% |
| uav0000137_00458_v | 233 | 3.90% | 30.54% |
| uav0000182_00000_v | 363 | -11.60% | 14.57% |
| uav0000268_05773_v | 978 | 0.00% | 0.00% |
| uav0000305_00000_v | 184 | -29.26% | 10.88% |
| uav0000339_00001_v | 275 | 19.13% | 18.85% |

**Total frames evaluated: 2,846**

**Evaluation note:** These are preliminary results obtained using a simplified pedestrian-only tracking evaluation protocol. They are not official VisDrone benchmark scores.

## Experimental Analysis

### Confidence Threshold Experiment

Lower confidence thresholds produced more pedestrian detection candidates in difficult aerial scenes.

However, additional detections did not necessarily result in successful tracks.

### Image Resolution Experiment

| Resolution | Pedestrian Detections |
|---|---:|
| 640 | 41 |
| 960 | 119 |
| 1280 | 207 |

These counts were measured over 100 frames of a selected VisDrone sequence using confidence 0.05.

The results demonstrate that increasing inference resolution can improve detection sensitivity for small objects.

However, detection counts alone do not establish detection accuracy.

### ByteTrack Optimization

Experiments with lower tracker thresholds recovered a limited number of pedestrian tracking observations.

The results suggest that tracker configuration influences tracking behavior, although accurate and consistent tracking remains challenging.

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Pradyumna11/Image_Detectives.git
cd Image_Detectives
```

### 2. Create a Virtual Environment

```bash
python3 -m venv .venv
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

## Model Weights and Dataset

The trained YOLOv8 model weights and VisDrone datasets are not included in this repository because of their size.

To run the application with the custom-trained model, place the weights at:

```text
runs/detect/runs/train/visdrone_full/weights/best.pt
```

Dataset-dependent evaluation scripts also require the appropriate VisDrone datasets and annotations.

## Running the Streamlit Dashboard

Start the application:

```bash
python -m streamlit run dashboard.py
```

Then open the local URL displayed by Streamlit.

The dashboard includes:

- Overview
- Image Detection
- Video Tracking
- Performance Evaluation

## Project Structure

```text
Image_Detectives/
|
|-- dashboard.py
|-- ByteTrack.py
|-- train.py
|-- train_full.py
|-- predict.py
|-- export_tracking.py
|-- evaluate_tracking.py
|-- evaluate_multiple_videos.py
|-- final_evaluation.py
|
|-- final_results/
|   |-- detection_summary.csv
|   |-- tracking_summary.csv
|   |-- confidence_summary.csv
|   |-- resolution_summary.csv
|   |-- bytetrack_summary.csv
|
|-- requirements.txt
|-- README.md
|-- .gitignore
```

## Project Limitations

- Small objects remain difficult to detect in aerial imagery.
- Detection accuracy varies between scenes.
- Low-confidence pedestrian detections may not initialize stable tracks.
- Higher inference resolution can reduce processing speed.
- The tracking evaluation uses a simplified pedestrian-only protocol.
- The trained model and datasets must be obtained separately.

## Future Improvements

- Improve detection accuracy for small objects.
- Investigate higher-resolution training and inference.
- Optimize ByteTrack parameters.
- Implement comprehensive multiclass tracking evaluation.
- Validate results against the official benchmark protocol.
- Extend the application to additional datasets.

## Author

**Pradyumna**


## Acknowledgments

This project uses the Ultralytics YOLO framework, ByteTrack, and the publicly available VisDrone dataset.
