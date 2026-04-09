# S.P.A.C.E - Smart Parking Application for Circulation Efficiency with TF OD API
> Parking lot occupancy detection (PKLot + custom dataset) using the TensorFlow Object Detection API.
<p align="center">
  <img src="assests/parkingreadme.png" alt="Smart parking application illustration" width="900">
</p>
---

## Table of Contents
- [Project Overview](#project-overview)
- [Repo Structure](#repo-structure)
- [Setup — Local venv (recommended)](#setup--local-venv-recommended)
- [Setup — Docker (for full reproducibility)](#setup--docker)
- [Datasets](#datasets)
- [Experiments](#experiments)
- [Running Notebooks](#running-notebooks)
- [Results](#results)

---

## Project Overview

This thesis investigates object detection for parking lot occupancy classification using:
- **PKLot** — a public benchmark dataset of parking lot images
- **Custom dataset** — collected and annotated manually

Models are trained using the **TensorFlow Object Detection API** and exported in multiple formats (SavedModel, TFLite, ONNX) for deployment analysis.


## Project Overview

This thesis investigates parking lot occupancy detection using two datasets and two training strategies:

- **PKLot** — public benchmark dataset of parking lot images
- **TelitlLot** — custom dataset collected and annotated manually
- **Labels:** `space-empty`, `space-occupied`
- **Model:** SSD MobileNet V2 FPN Lite 320x320 (pretrained on COCO17, fine-tuned)
- **Strategies:** Fine-tuning vs Frozen Backbone
- **Exports:** SavedModel, TFLite (float32/float16/int8).
---

## Repo Structure

```
SPACE/
├── configs/
│   ├── pklot/pipeline.config        ← TF OD API pipeline for PKLot
│   └── custom/pipeline.config       ← TF OD API pipeline for custom dataset
├── datasets/
│   ├── pklot/                        ← gitignored (see Datasets section)
│   │   ├── train/  (train.tfrecord + trainlabel_map.pbtxt)
│   │   ├── valid/  (valid.tfrecord + validlabel_map.pbtxt)
│   │   └── test/   (test.tfrecord  + testlabel_map.pbtxt)
│   └── custom/                       ← same structure as pklot
├── models/                           ← gitignored (checkpoints + exports)
├── notebooks/
│   ├── 03_training_pklot.ipynb       ← E1 (fine-tune) + E3 (frozen backbone)
│   ├── 04_training_custom.ipynb      ← E5 + E6
│   ├── 05_evaluation.ipynb           ← E1, E3, E5, E6
│   ├── 06_cross_dataset_eval.ipynb   ← E2, E4 (cross-domain)
│   ├── 07_export.ipynb               ← SavedModel + TFLite + ONNX
│   └── 08_inference_demo.ipynb       ← visual demo + model comparison
├── results/                          ← metrics JSON + plots
├── scripts/
│   ├── experiment_configs.py         ← all paths + experiment definitions
│   ├── patch_config.py               ← auto-patches pipeline config per experiment
│   └── setup/
│       └── download_base_model.sh    ← downloads COCO17 pretrained checkpoint
├── tensorflow_models/                ← git submodule (tensorflow/models)
├── setup_local.py                    ← automated local setup script
├── Dockerfile                        ← for Docker-based reproducibility
├── docker-compose.yml
└── requirements.txt                  ← pinned: tf==2.11.0, protobuf==3.19.6
```

---

## Setup — Local venv (recommended)

> Requires **Python 3.7** and **Windows**. One script does everything.

### 1. Clone with submodule
```powershell
git clone --recurse-submodules https://github.com/poorni666/Automated-Vision-system-for-Smart-parking-efficiency.git
cd SPACE

# If you forgot --recurse-submodules:
git submodule update --init --recursive
```

### 2. Create virtual environment
```powershell
py -3.7 -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Run automated setup (one command does everything)
```powershell
python setup_local.py
```

This automatically:
- Installs all pinned requirements
- Downloads `protoc.exe` v3.19.6
- Compiles TF OD API protobuf definitions
- Installs the `object_detection` package
- Permanently adds `PYTHONPATH` to your venv

### 4. Verify
```powershell
python -c "import tensorflow as tf; print(tf.__version__)"
# 2.11.0

python -c "from object_detection.utils import label_map_util; print('OD API OK')"
# OD API OK
```

### Daily workflow
```powershell
cd SPACE
.venv\Scripts\Activate.ps1
jupyter notebook notebooks/
```

---

## Setup — Docker

> Use this for full environment reproducibility on any OS.

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (GPU only)
- Git


### 1. Clone with submodule
```bash
git clone --recurse-submodules https://github.com/poorni666/Automated-Vision-system-for-Smart-parking-efficiency.git
cd SPACE
git submodule update --init --recursive
```

### 2. Build and start
```bash
# GPU
docker compose up --build

# CPU only
docker compose -f docker-compose.yml -f docker-compose.cpu.yml up --build
```

### 3. Open Jupyter
- Jupyter → **http://localhost:8888**
- TensorBoard → **http://localhost:6006**

---
## Datasets

### PKLot
PKLot is not included in this repo due to size. Download it from:
- **Kaggle**: https://www.kaggle.com/datasets/ammarnassanalhajali/pklot-dataset
- **Original paper site**: https://web.inf.ufpr.br/vri/databases/parking-lot-database/

Place images at: `datasets/pklot/raw/`

### Custom Dataset
The custom dataset annotations are in `datasets/custom/annotations/` (Pascal VOC XML format).
Raw images are not included — contact the author or see the thesis for acquisition details.

### Generate TFRecords
After placing images in the correct folders:
```bash
python scripts/dataset_prep/convert_to_tfrecord.py --dataset pklot
python scripts/dataset_prep/convert_to_tfrecord.py --dataset custom
```
---

## Experiments

| Exp | Model | Train | Strategy | Test | Type |
|-----|-------|-------|----------|------|------|
| E1 | M1 | PKLot | Fine-Tuning | PKLot | In-Domain |
| E2 | M1 | PKLot | Fine-Tuning | TelitlLot | Cross-Domain |
| E3 | M2 | PKLot | Frozen Backbone | PKLot | In-Domain |
| E4 | M2 | PKLot | Frozen Backbone | TelitlLot | Cross-Domain |
| E5 | M3 | TelitlLot | Fine-Tuning | TelitlLot | Target-Domain |
| E6 | M4 | TelitlLot | Frozen Backbone | TelitlLot | Target-Domain |

Experiments are configured in `scripts/experiment_configs.py` — no manual path editing needed.

---

## Running Notebooks

```powershell
# Activate venv first
.venv\Scripts\Activate.ps1

# Launch Jupyter
jupyter notebook notebooks/
```

Run in order:

| Notebook | What it does |
|----------|-------------|
| `03_training_pklot.ipynb` | Trains E1 + E3 |
| `04_training_custom.ipynb` | Trains E5 + E6 |
| `05_evaluation.ipynb` | Evaluates E1, E3, E5, E6 |
| `06_cross_dataset_eval.ipynb` | Evaluates E2, E4 (cross-domain) |
| `07_export.ipynb` | Exports SavedModel, TFLite, ONNX |
| `08_inference_demo.ipynb` | Visual demo + speed benchmark |

---
## Model Architecture

The model used in this project is **SSD MobileNet V2 FPN Lite 320×320**, a lightweight single-stage object detector pretrained on COCO 2017 and fine-tuned for parking space occupancy detection.

It was selected because it strikes the right balance between speed and accuracy for real-time parking monitoring — small enough to run on edge hardware, accurate enough for reliable occupancy classification.

### Components

**Backbone — MobileNet V2**
Extracts visual features from the input image using depthwise separable convolutions, which are significantly cheaper to compute than standard convolutions. This is what makes the model lightweight. The backbone is pretrained on COCO 2017, meaning it already understands general visual concepts (edges, shapes, textures) before we ever show it a parking lot.

**Neck — FPN Lite (Feature Pyramid Network)**
Parking lot images are challenging because spaces appear at very different scales ,a space close to the camera looks large, while one far away looks tiny. FPN Lite solves this by combining feature maps from multiple backbone levels, creating a multi-scale representation that handles both large and small spaces in the same image.

**Head — Convolutional Predictor**
Takes the fused feature maps and outputs bounding boxes + class scores for each anchor. The same weights are shared across all scales, which keeps the model compact and helps it generalise.

**Post-processing — NMS**
Filters overlapping detections, keeping only the most confident prediction per parking space.

### Training Strategies

Two strategies are compared across the 6 experiments:

**Fine-Tuning**
All weights — backbone, neck, and head — are updated during training. The backbone adapts from COCO features to parking-specific features. This generally achieves higher accuracy.

**Frozen Backbone**
Only the neck and head are trained. The backbone stays fixed at its COCO pretrained weights. This tests how well general visual features transfer to the parking domain without any adaptation. Training is faster and the model relies entirely on transfer learning for feature extraction.


## Results

| Dataset | Model | mAP@0.5 | Inference (ms) |
|---------|-------|---------|----------------|
| PKLot   | SSD MobileNetV2 | — | — |
| Custom  | SSD MobileNetV2 | — | — |

> Results will be populated after training. See `results/metrics/` for full evaluation output.

---

## Notebooks

| # | Notebook | Description |
|---|----------|-------------|
| 01 | `dataset_exploration.ipynb` | EDA on PKLot and custom dataset |
| 02 | `data_preparation.ipynb` | Annotation parsing, TFRecord generation |
| 03 | `training_pklot.ipynb` | Training walkthrough — PKLot |
| 04 | `training_custom.ipynb` | Training walkthrough — Custom dataset |
| 05 | `evaluation.ipynb` | mAP, precision/recall, confusion matrix |
| 06 | `inference_demo.ipynb` | Live inference with visualized detections |

---

## Citation

If you use this work, please cite the thesis:
```
[Your Name], "[Thesis Title]", [University], [Year].
```
