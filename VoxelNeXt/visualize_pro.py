import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d.art3d import Line3DCollection

with open("output/kitti_models/voxelnext_cones/default/eval/epoch_50/val/default/result.pkl", 'rb') as f:
    predictions = pickle.load(f)
with open("/mnt/c/Users/Student/Desktop/VoxelNext_V2/data/kitti_cones/kitti_infos_val.pkl", 'rb') as f:
    gt_infos = pickle.load(f)

def get_corners(box):
    x, y, z, dx, dy, dz, rot = box
    corners = np.array([
        [ dx/2,  dy/2, -dz/2], [-dx/2,  dy/2, -dz/2],
        [-dx/2, -dy/2, -dz/2], [ dx/2, -dy/2, -dz/2],
        [ dx/2,  dy/2,  dz/2], [-dx/2,  dy/2,  dz/2],
        [-dx/2, -dy/2,  dz/2], [ dx/2, -dy/2,  dz/2],
    ])
    c, s = np.cos(rot), np.sin(rot)
    R = np.array([[c,-s,0],[s,c,0],[0,0,1]])
    return (R @ corners.T).T + [x, y, z]

def bev_iou(pb, gb):
    ix = max(0, min(pb[0]+pb[3]/2, gb[0]+gb[3]/2) - max(pb[0]-pb[3]/2, gb[0]-gb[3]/2))
    iy = max(0, min(pb[1]+pb[4]/2, gb[1]+gb[4]/2) - max(pb[1]-pb[4]/2, gb[1]-gb[4]/2))
    inter = ix * iy
    union = pb[3]*pb[4] + gb[3]*gb[4] - inter
    return inter / union if union > 0 else 0

def classify_preds(pred_boxes, pred_scores, gt_boxes, iou_thresh=0.5):
    """Klassifiziere jede Prediction als TP oder FP"""
    matched_gt = set()
    labels = []
    for k, pb in enumerate(pred_boxes):
        if len(gt_boxes) == 0:
            labels.append('FP')
            continue
        ious = [bev_iou(pb, gb) for gb in gt_boxes]
        best_gi = int(np.argmax(ious))
        if ious[best_gi] >= iou_thresh and best_gi not in matched_gt:
            labels.append('TP')
            matched_gt.add(best_gi)
        else:
            labels.append('FP')
    # FN = GT die nicht gematcht wurden
    fn_indices = [i for i in range(len(gt_boxes)) if i not in matched_gt]
    return labels, fn_indices

COLORS = {
    'points': '#2d2d2d',
    'TP': '#00cc44',      # grün
    'FP': '#ff3333',      # rot
    'FN': '#ff9900',      # orange
    'GT': '#4488ff',      # blau
}

for FRAME_IDX in range(len(predictions)):
    pred = predictions[FRAME_IDX]
    gt_info = gt_infos[FRAME_IDX]
    lidar_idx = gt_info['point_cloud']['lidar_idx']

    points = np.fromfile(
        f"/mnt/c/Users/Student/Desktop/VoxelNext_V2/data/kitti_cones/training/velodyne/{lidar_idx}.bin",
        dtype=np.float32).reshape(-1, 4)[:, :3]

    gt_boxes = gt_info['annos']['gt_boxes_lidar']
    mask = pred['score'] >= 0.5
    pred_boxes = pred['boxes_lidar'][mask]
    pred_scores = pred['score'][mask]

    pred_labels, fn_indices = classify_preds(pred_boxes, pred_scores, gt_boxes)

    # Zoom: nur Bereich um die Cones
    all_centers = np.vstack([gt_boxes[:, :2], pred_boxes[:, :2]]) if len(pred_boxes) > 0 else gt_boxes[:, :2]
    cx, cy = all_centers[:, 0].mean(), all_centers[:, 1].mean()
    margin = 8

    fig = plt.figure(figsize=(18, 8), facecolor='white')
    fig.suptitle(f'VoxelNeXt — Frame {FRAME_IDX} (ID: {lidar_idx})', fontsize=14, fontweight='bold')

    # === BEV FULL ===
    ax1 = fig.add_subplot(131)
    ax1.set_facecolor('#f8f8f8')
    ax1.scatter(points[:,0], points[:,1], s=0.2, c=COLORS['points'], alpha=0.3)
    for box in gt_boxes:
        c = get_corners(box)
        bev = c[[0,1,2,3,0], :2]
        ax1.plot(bev[:,0], bev[:,1], color=COLORS['GT'], linewidth=1.2)
    for i, box in enumerate(pred_boxes):
        c = get_corners(box)
        bev = c[[0,1,2,3,0], :2]
        col = COLORS[pred_labels[i]]
        ax1.plot(bev[:,0], bev[:,1], color=col, linewidth=1.5)
    ax1.set_xlim(-35,35); ax1.set_ylim(-35,35)
    ax1.set_xlabel('X (m)'); ax1.set_ylabel('Y (m)')
    ax1.set_title('BEV (Übersicht)', fontsize=11)
    ax1.set_aspect('equal'); ax1.grid(True, alpha=0.3)

    # === BEV ZOOM ===
    ax2 = fig.add_subplot(132)
    ax2.set_facecolor('#f0f4ff')
    # Punkte im Zoom-Bereich
    pm = (points[:,0] > cx-margin) & (points[:,0] < cx+margin) & \
         (points[:,1] > cy-margin) & (points[:,1] < cy+margin)
    ax2.scatter(points[pm,0], points[pm,1], s=1.5, c=COLORS['points'], alpha=0.5)

    # GT Boxen (blau)
    for i, box in enumerate(gt_boxes):
        c = get_corners(box)
        bev = c[[0,1,2,3,0], :2]
        is_fn = i in fn_indices
        col = COLORS['FN'] if is_fn else COLORS['GT']
        ax2.plot(bev[:,0], bev[:,1], color=col, linewidth=2.5, linestyle='--' if is_fn else '-')
        ax2.text(box[0], box[1]+0.15, 'FN' if is_fn else 'GT', color=col, fontsize=7, ha='center', fontweight='bold')

    # Pred Boxen (TP=grün, FP=rot)
    for i, box in enumerate(pred_boxes):
        c = get_corners(box)
        bev = c[[0,1,2,3,0], :2]
        col = COLORS[pred_labels[i]]
        ax2.plot(bev[:,0], bev[:,1], color=col, linewidth=2.5)
        ax2.text(box[0], box[1]-0.25, f'{pred_labels[i]}\n{pred_scores[i]:.2f}',
                 color=col, fontsize=6.5, ha='center', fontweight='bold')

    ax2.set_xlim(cx-margin, cx+margin); ax2.set_ylim(cy-margin, cy+margin)
    ax2.set_xlabel('X (m)'); ax2.set_ylabel('Y (m)')
    tp_count = pred_labels.count('TP')
    fp_count = pred_labels.count('FP')
    fn_count = len(fn_indices)
    ax2.set_title(f'Zoom | TP={tp_count}  FP={fp_count}  FN={fn_count}', fontsize=11)
    ax2.set_aspect('equal'); ax2.grid(True, alpha=0.3)

    legend = [
        mpatches.Patch(color=COLORS['GT'],  label=f'GT ({len(gt_boxes)})'),
        mpatches.Patch(color=COLORS['TP'],  label=f'TP ({tp_count})'),
        mpatches.Patch(color=COLORS['FP'],  label=f'FP ({fp_count})'),
        mpatches.Patch(color=COLORS['FN'],  label=f'FN ({fn_count})'),
    ]
    ax2.legend(handles=legend, loc='upper right', fontsize=8)

    # === 3D VIEW ===
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.set_facecolor('#f8f8f8')
    ax3.scatter(points[pm,0], points[pm,1], points[pm,2], s=0.5, c='#888888', alpha=0.4)

    lines_idx = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]]
    for i, box in enumerate(gt_boxes):
        corners = get_corners(box)
        col = COLORS['FN'] if i in fn_indices else COLORS['GT']
        segs = [[corners[a], corners[b]] for a,b in lines_idx]
        ax3.add_collection3d(Line3DCollection(segs, colors=col, linewidths=2))
    for i, box in enumerate(pred_boxes):
        corners = get_corners(box)
        col = COLORS[pred_labels[i]]
        segs = [[corners[a], corners[b]] for a,b in lines_idx]
        ax3.add_collection3d(Line3DCollection(segs, colors=col, linewidths=2))

    ax3.set_xlim(cx-margin, cx+margin)
    ax3.set_ylim(cy-margin, cy+margin)
    ax3.set_zlim(-1, 2)
    ax3.set_xlabel('X'); ax3.set_ylabel('Y'); ax3.set_zlabel('Z')
    ax3.set_title('3D View', fontsize=11)
    ax3.view_init(elev=20, azim=-60)

    plt.tight_layout()
    out = f"/mnt/c/Users/Student/Desktop/VoxelNext_V2/viz_pro_frame_{FRAME_IDX:02d}_{lidar_idx}.png"
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Frame {FRAME_IDX:02d} | TP={tp_count} FP={fp_count} FN={fn_count} | {out}")

print("\n🎉 Alle Frames gespeichert!")
