import pickle
import numpy as np
from pathlib import Path

# === Lade Predictions ===
result_path = "output/kitti_models/voxelnext_cones/default/eval/epoch_50/val/default/result.pkl"
with open(result_path, 'rb') as f:
    predictions = pickle.load(f)

# === Lade Ground Truth ===
gt_path = "/mnt/c/Users/Student/Desktop/VoxelNext_V2/data/kitti_cones/kitti_infos_val.pkl"
with open(gt_path, 'rb') as f:
    gt_infos = pickle.load(f)

print(f"Predictions: {len(predictions)} frames")
print(f"GT frames: {len(gt_infos)}")
print(f"GT keys: {gt_infos[0].keys()}")
print(f"GT annos keys: {gt_infos[0]['annos'].keys()}")
