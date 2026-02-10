# gun-detection-


# Gun Detection

Object detection system for identifying firearms in images using YOLOv26n.

## Overview

This project implements real-time gun detection using the YOLOv26 nano model. The system can detect firearms in images and provides both a command-line interface and a web-based Gradio app for easy interaction.

## Dataset

Trained on the [Guns Object Detection dataset](https://www.kaggle.com/datasets/issaisasank/guns-object-detection) from Kaggle, containing labeled images of various firearms.

## Model Performance

Training completed over 100 epochs with the following results:

| Metric | Value |
|--------|-------|
| Precision | 0.759 |
| Recall | 0.692 |
| mAP@0.5 | 0.744 |
| mAP@0.5:0.95 | 0.387 |

The model was trained on 67 validation images containing 109 gun instances.

## Project Structure

```
gun-detection/
├── data/
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── val/
│   │   ├── images/
│   │   └── labels/
│   └── data.yaml
├── scripts/
│   ├── preprocess.py      # Data preprocessing utilities
│   ├── preprocessing.py   # Data splitting pipeline
│   ├── train.py          # Model training
│   ├── detect.py         # CLI detection tool
│   └── app.py            # Gradio web interface
├── notebooks/
│   └── yolo_model.ipynb  # Training notebook
├── runs/
│   └── detect/train/weights/
│       └── best.pt       # Trained model weights
├── pyproject.toml        # Project dependencies
└── uv.lock              # Locked dependencies
```

## Setup

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

### Installation

1. Clone the repository:
```bash
git clone https://github.com/lonelyalpaca2003/gun-detection.git
cd gun-detection
```

2. Install dependencies:
```bash
uv sync
```

3. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/issaisasank/guns-object-detection) and place it in the `data/` directory.

## Usage

### Preprocessing Data

Convert raw annotations to YOLO format and split into train/val sets:

```bash
uv run python scripts/preprocessing.py
```

### Training

Train the model from scratch:

```bash
uv run python scripts/train.py
```

Training parameters:
- Model: YOLOv26n
- Epochs: 100
- Batch size: 16
- Device: Auto-detected (CUDA/MPS/CPU)

### Detection (CLI)

Run detection on a single image:

```bash
uv run python scripts/detect.py path/to/image.jpg
```

Results are saved to `result.jpg`.

### Web Interface

Launch the Gradio app for interactive detection:

```bash
uv run python scripts/app.py
```

Upload an image through the web interface and adjust the confidence threshold to control detection sensitivity.

## Model Architecture

YOLOv26 nano is a lightweight object detection model optimized for speed and efficiency:
- 122 layers
- 2.4M parameters
- 5.2 GFLOPs

## Technical Stack

- **Model**: YOLOv26n (Ultralytics)
- **Framework**: PyTorch
- **Deployment**: Gradio
- **Image Processing**: OpenCV, PIL
- **Dependency Management**: uv

## Dependencies

Key dependencies (see `pyproject.toml` for full list):
- ultralytics
- torch
- opencv-python
- gradio
- numpy
- matplotlib

## License

[Add your license here]

## Acknowledgments

Dataset: [Guns Object Detection](https://www.kaggle.com/datasets/issaisasank/guns-object-detection) by Issai Sasank
