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

def evaluate(iou_thresh, score_thresh):
    all_scores, all_tp = [], []
    total_gt = 0
    for pred, gt_info in zip(predictions, gt_infos):
        gt_boxes = gt_info['annos']['gt_boxes_lidar']
        gt_names = gt_info['annos']['name']
        gt_boxes = gt_boxes[gt_names == 'small_cone']
        total_gt += len(gt_boxes)

        pred_boxes = pred['boxes_lidar']
        pred_scores = pred['score']
        mask = pred_scores >= score_thresh
        pred_boxes = pred_boxes[mask]
        pred_scores = pred_scores[mask]

        order = np.argsort(-pred_scores)
        pred_boxes = pred_boxes[order]
        pred_scores = pred_scores[order]

        matched_gt = set()
        for k in range(len(pred_boxes)):
            all_scores.append(pred_scores[k])
            if len(gt_boxes) == 0:
                all_tp.append(0)
                continue
            ious = [bev_iou(pred_boxes[k], gb) for gb in gt_boxes]
            best_gi = int(np.argmax(ious))
            if ious[best_gi] >= iou_thresh and best_gi not in matched_gt:
                all_tp.append(1)
                matched_gt.add(best_gi)
            else:
                all_tp.append(0)

    if len(all_scores) == 0:
        return dict(ap=0, precision=0, recall=0, f1=0, tp=0, fp=0, fn=total_gt, preds=0, gt=total_gt)

    all_tp_arr = np.array(all_tp)
    order = np.argsort(-np.array(all_scores))
    all_tp_arr = all_tp_arr[order]
    cum_tp = np.cumsum(all_tp_arr)
    cum_fp = np.cumsum(1 - all_tp_arr)
    precisions = cum_tp / (cum_tp + cum_fp + 1e-6)
    recalls = cum_tp / (total_gt + 1e-6)
    ap = compute_ap(recalls, precisions)

    tp = int(all_tp_arr.sum())
    fp = len(all_tp_arr) - tp
    fn = total_gt - tp
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (total_gt + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    return dict(ap=ap*100, precision=precision*100, recall=recall*100,
                f1=f1*100, tp=tp, fp=fp, fn=fn, preds=len(all_scores), gt=total_gt)

# === Header ===
hdr = f"{'':>6} | {'AP':>7} | {'Prec':>7} | {'Recall':>7} | {'F1':>7} | {'TP':>5} | {'FP':>5} | {'FN':>5} | {'Preds':>6}"
sep = "-" * len(hdr)

# === IoU Threshold Analyse ===
print("\n📊 IoU Threshold Analyse (Score >= 0.5 fix):")
print(hdr.replace("     ", "IoU  ", 1))
print(sep)
for iou in [0.25, 0.30, 0.40, 0.50, 0.60, 0.70]:
    r = evaluate(iou, 0.5)
    print(f"{iou:>6.2f} | {r['ap']:>6.2f}% | {r['precision']:>6.2f}% | {r['recall']:>6.2f}% | {r['f1']:>6.2f}% | {r['tp']:>5} | {r['fp']:>5} | {r['fn']:>5} | {r['preds']:>6}")

# === Score Threshold Analyse ===
print(f"\n📊 Score Threshold Analyse (IoU >= 0.5 fix)  [Total GT: {evaluate(0.5,0.0)['gt']}]:")
print(hdr.replace("     ", "Score", 1))
print(sep)
for score in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    r = evaluate(0.5, score)
    print(f"{score:>6.2f} | {r['ap']:>6.2f}% | {r['precision']:>6.2f}% | {r['recall']:>6.2f}% | {r['f1']:>6.2f}% | {r['tp']:>5} | {r['fp']:>5} | {r['fn']:>5} | {r['preds']:>6}")
