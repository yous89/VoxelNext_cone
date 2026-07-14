# VoxelNeXt for Traffic Cone Detection

This repository contains an adapted version of **VoxelNeXt (CVPR 2023)** for real-time detection of small traffic cones in LiDAR point clouds.

The project was developed as part of a Master's thesis at Hochschule Darmstadt for autonomous Formula Student racing.

## Features

- VoxelNeXt adapted for traffic cone detection
- KITTI-format dataset
- Ouster OS2-128 support
- Real-time inference
- BEV demo visualization
- Confidence scores
- LiDAR point statistics
- Model FPS and Pipeline FPS

---

# Installation

Clone the repository

```bash
git clone https://github.com/yous89/VoxelNext_cone.git
cd VoxelNext_cone/VoxelNeXt
```

Create environment

```bash
conda create -n voxelnext python=3.8
conda activate voxelnext
```

Install requirements

```bash
pip install -r requirements.txt
```

Install OpenPCDet and spconv according to the official documentation.

---

# Dataset

Dataset structure

```
data/
└── kitti_cones/
    ├── ImageSets/
    ├── training/
    │   ├── velodyne/
    │   ├── label_2/
    │   └── calib/
    ├── kitti_infos_train.pkl
    ├── kitti_infos_val.pkl
    └── kitti_dbinfos_train.pkl
```

---

# Training

Go to

```bash
cd tools
```

Train the model

```bash
python train.py \
    --cfg_file cfgs/kitti_models/voxelnext_cones.yaml \
    --batch_size 4
```

Resume training

```bash
python train.py \
    --cfg_file cfgs/kitti_models/voxelnext_cones.yaml \
    --ckpt ../output/kitti_models/voxelnext_cones/default/ckpt/checkpoint_epoch_50.pth
```

---

# Evaluation

Evaluate a checkpoint

```bash
cd tools

python test.py \
    --cfg_file cfgs/kitti_models/voxelnext_cones.yaml \
    --ckpt ../output/kitti_models/voxelnext_cones/default/ckpt/checkpoint_epoch_50.pth
```

---
# Detection Results

<p align="center">
<img src="VoxelNeXt/output/kitti_models/voxelnext_cones/sr_training/ckpt/Screenshot 2026-07-08 134832.png" width="950">
</p>
---

# Ouster PCAP Demo

Run the BEV demo

```bash
cd tools

python demo_ouster_voxelnext_bev.py \
    --cfg_file cfgs/kitti_models/voxelnext_cones.yaml \
    --ckpt ../output/kitti_models/voxelnext_cones/default/ckpt/checkpoint_epoch_50.pth \
    --pcap "/path/to/recording.pcap" \
    --metadata "/path/to/metadata.json" \
    --score_thresh 0.5 \
    --x_min -18 \
    --x_max 18 \
    --y_min -18 \
    --y_max 18 \
    --output ../demo_video/voxelnext_demo.mp4
```

The demo visualizes

- Bird's-Eye View
- Bounding boxes
- Confidence score
- Number of LiDAR points inside each detection
- Model FPS
- Pipeline FPS

---

# Results

Example checkpoint

```
output/kitti_models/
└── voxelnext_cones/
    └── default/
        └── ckpt/
            └── checkpoint_epoch_48.pth
```

---

# Citation

If you use this repository, please cite

```
@inproceedings{chen2023voxelnext,
  title={VoxelNeXt: Fully Sparse VoxelNet for 3D Object Detection and Tracking},
  author={Yukang Chen et al.},
  booktitle={CVPR},
  year={2023}
}
```

---

# Acknowledgement

This repository is based on

- VoxelNeXt
- OpenPCDet
- spconv

- # Results

The final VoxelNeXt model was trained and evaluated on the custom **KITTI Cone Dataset**.

| Metric | Value |
|---------|------:|
| Precision | **89.1%** |
| Recall | **84.6%** |
| F1-score | **86.8%** |
| AP@0.5 | **72.7%** |
| Inference Speed | **55 FPS** |
| Inference Time | **18.2 ms** |
| GPU Memory | **2.5 GB** |
| Real-time Capable | ✅ |

---

## Distance-based Recall

| Distance | Recall |
|-----------|------:|
| 0–5 m | **93.0%** |
| 5–10 m | **92.0%** |
| 10–15 m | **64.8%** |
| 15–20 m | **20.0%** |

---

## Demo Features

- ✅ Ouster OS2-128 support
- ✅ Bird's-Eye View (BEV) visualization
- ✅ Real-time cone detection
- ✅ Confidence score for each detection
- ✅ Number of LiDAR points per detected object
- ✅ Model FPS
- ✅ Pipeline FPS
- ✅ Video export (.mp4)


## Training Configuration

| Parameter | Value |
|-----------|-------|
| Model | VoxelNeXt |
| Dataset | Custom KITTI Cone Dataset |
| Classes | small_cone |
| Input | LiDAR Point Cloud (x, y, z, reflectivity) |
| Sensor | Ouster OS2-128 |
| Epochs | 48 |
| Optimizer | Adam OneCycle |
| Framework | OpenPCDet |
| CUDA | Supported |
