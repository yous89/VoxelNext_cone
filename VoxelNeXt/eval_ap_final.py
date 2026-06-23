import pickle
import numpy as np

with open("output/kitti_models/voxelnext_cones/default/eval/epoch_50/val/default/result.pkl", 'rb') as f:
    predictions = pickle.load(f)

with open("/mnt/c/Users/Student/Desktop/VoxelNext_V2/data/kitti_cones/kitti_infos_val.pkl", 'rb') as f:
    gt_infos = pickle.load(f)

def bev_iou(pb, gb):
    ix = max(0, min(pb[0]+pb[3]/2, gb[0]+gb[3]/2) - max(pb[0]-pb[3]/2, gb[0]-gb[3]/2))
    iy = max(0, min(pb[1]+pb[4]/2, gb[1]+gb[4]/2) - max(pb[1]-pb[4]/2, gb[1]-gb[4]/2))
    inter = ix * iy
    union = pb[3]*pb[4] + gb[3]*gb[4] - inter
    return inter / union if union > 0 else 0

def compute_ap(recalls, precisions):
    ap = 0
    for t in np.arange(0, 1.1, 0.1):
        p = precisions[recalls >= t]
        ap += (np.max(p) if len(p) > 0 else 0)
    return ap / 11.0

all_scores, all_tp = [], []
total_gt = 0

for pred, gt_info in zip(predictions, gt_infos):
    gt_boxes = gt_info['annos']['gt_boxes_lidar']
    gt_names = gt_info['annos']['name']
    gt_boxes = gt_boxes[gt_names == 'small_cone']
    total_gt += len(gt_boxes)

    pred_boxes = pred['boxes_lidar']
    pred_scores = pred['score']
    order = np.argsort(-pred_scores)
    pred_boxes = pred_boxes[order]
    pred_scores = pred_scores[order]

    matched_gt = set()
    for k in range(len(pred_boxes)):
        all_scores.append(pred_scores[k])
        if len(gt_boxes) == 0:
            all_tp.append(0)
            continue
        ious = [bev_iou(pred_boxes[k], gb) for gi, gb in enumerate(gt_boxes)]
        best_gi = int(np.argmax(ious))
        if ious[best_gi] >= 0.5 and best_gi not in matched_gt:
            all_tp.append(1)
            matched_gt.add(best_gi)
        else:
            all_tp.append(0)

all_scores = np.array(all_scores)
all_tp = np.array(all_tp)
order = np.argsort(-all_scores)
all_tp = all_tp[order]
cum_tp = np.cumsum(all_tp)
cum_fp = np.cumsum(1 - all_tp)
precisions = cum_tp / (cum_tp + cum_fp + 1e-6)
recalls = cum_tp / (total_gt + 1e-6)
ap = compute_ap(recalls, precisions)

print("=" * 50)
print(f"  Total GT:          {total_gt}")
print(f"  Total Preds:       {len(all_scores)}")
print(f"  TP (@IoU>=0.5):    {int(all_tp.sum())}")
print(f"  Max Recall:        {recalls[-1]:.4f}")
print(f"  AP@0.5 (BEV):      {ap*100:.2f}%")
print("=" * 50)
