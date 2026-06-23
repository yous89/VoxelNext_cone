import pickle
import numpy as np

result_path = "output/kitti_models/voxelnext_cones/default/eval/epoch_50/val/default/result.pkl"
with open(result_path, 'rb') as f:
    results = pickle.load(f)

print(f"Anzahl Frames: {len(results)}")
print(f"Keys im ersten Frame: {results[0].keys()}")
print(f"Beispiel: {results[0]}")
