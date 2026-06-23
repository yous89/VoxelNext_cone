import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.colors import LinearSegmentedColormap

with open("output/kitti_models/voxelnext_cones/default/eval/epoch_50/val/default/result.pkl", 'rb') as f:
    predictions = pickle.load(f)
with open("/mnt/c/Users/Student/Desktop/VoxelNext_V2/data/kitti_cones/kitti_infos_val.pkl", 'rb') as f:
    gt_infos = pickle.load(f)

DATA_ROOT = "/mnt/c/Users/Student/Desktop/VoxelNext_V2/data/kitti_cones"

# Benutzerdefinierte Colormap:
# Boden (Z < 0.1m) = dunkelgrau
# Cones (Z > 0.1m) = hellgelb → orange → rot
cone_cmap = LinearSegmentedColormap.from_list('cone_map', [
    (0.0,  '#2c2c3e'),   # Boden: dunkelgrau/blau
    (0.12, '#3a3a50'),   # Boden oberfläche
    (0.18, '#f0e040'),   # Cone Basis: gelb
    (0.55, '#ff8800'),   # Cone Mitte: orange
    (1.0,  '#ff2200'),   # Cone Spitze: rot
])

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

def classify_preds(pred_boxes, gt_boxes, iou_thresh=0.5):
    matched_gt = set()
    labels = []
    for pb in pred_boxes:
        if len(gt_boxes) == 0:
            labels.append('FP'); continue
        ious = [bev_iou(pb, gb) for gb in gt_boxes]
        best_gi = int(np.argmax(ious))
        if ious[best_gi] >= iou_thresh and best_gi not in matched_gt:
            labels.append('TP'); matched_gt.add(best_gi)
        else:
            labels.append('FP')
    fn_indices = [i for i in range(len(gt_boxes)) if i not in matched_gt]
    return labels, fn_indices

def draw_bev_box(ax, box, color, lw=2.5, linestyle='-', label=None, score=None):
    c = get_corners(box)
    bev = c[[0,1,2,3,0], :2]
    ax.plot(bev[:,0], bev[:,1], color=color, linewidth=lw, linestyle=linestyle)
    if label:
        txt = label if score is None else f"{label}\n{score:.2f}"
        ax.text(box[0], box[1], txt, color=color, fontsize=8,
                ha='center', va='center', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                          alpha=0.75, edgecolor=color))

def count_points_in_box(points, box):
    x, y, z, dx, dy, dz, _ = box
    mask = (
        (points[:,0] >= x-dx/2) & (points[:,0] <= x+dx/2) &
        (points[:,1] >= y-dy/2) & (points[:,1] <= y+dy/2) &
        (points[:,2] >= z-dz/2) & (points[:,2] <= z+dz/2)
    )
    return int(mask.sum())

def setup_3d_ax(ax, xlim, ylim, zlim, azim, elev=20):
    ax.set_facecolor('#0a0a1a')
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_zlim(*zlim)
    ax.set_xlabel('X (m)', fontsize=9, color='white')
    ax.set_ylabel('Y (m)', fontsize=9, color='white')
    ax.set_zlabel('Z (m)', fontsize=9, color='white')
    ax.tick_params(colors='white', labelsize=7)
    ax.view_init(elev=elev, azim=azim)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('#222244')
    ax.yaxis.pane.set_edgecolor('#222244')
    ax.zaxis.pane.set_edgecolor('#222244')

COLORS = {'TP': '#2ecc71', 'FP': '#e74c3c', 'FN': '#f39c12', 'GT': '#3498db'}
LINES  = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]]

for FRAME_IDX in range(len(predictions)):
    pred      = predictions[FRAME_IDX]
    gt_info   = gt_infos[FRAME_IDX]
    lidar_idx = gt_info['point_cloud']['lidar_idx']

    all_points = np.fromfile(f"{DATA_ROOT}/training/velodyne/{lidar_idx}.bin",
                             dtype=np.float32).reshape(-1, 4)[:, :3]
    total_pts  = len(all_points)

    gt_boxes    = gt_info['annos']['gt_boxes_lidar']
    mask        = pred['score'] >= 0.5
    pred_boxes  = pred['boxes_lidar'][mask]
    pred_scores = pred['score'][mask]
    pred_labels, fn_indices = classify_preds(pred_boxes, gt_boxes)

    tp = pred_labels.count('TP')
    fp = pred_labels.count('FP')
    fn = len(fn_indices)

    all_c    = np.vstack([gt_boxes[:,:2], pred_boxes[:,:2]]) if len(pred_boxes) > 0 else gt_boxes[:,:2]
    cx, cy   = all_c[:,0].mean(), all_c[:,1].mean()
    gt_dists = np.sqrt(gt_boxes[:,0]**2 + gt_boxes[:,1]**2)
    max_dist = gt_dists.max()
    margin   = float(np.clip(max_dist * 0.55 + 2.5, 5, 12))
    azim     = float(-np.degrees(np.arctan2(cy, cx)) - 90)

    dist_all = np.sqrt(all_points[:,0]**2 + all_points[:,1]**2)
    pm = ((all_points[:,0] > cx-margin) & (all_points[:,0] < cx+margin) &
          (all_points[:,1] > cy-margin) & (all_points[:,1] < cy+margin) &
          (all_points[:,2] > -0.3)      & (all_points[:,2] < 1.2) &
          (dist_all > 1.5))
    pts      = all_points[pm]
    zoom_pts = len(pts)

    # Z normalisiert für Colormap (0=Boden, 1=Cone-Spitze)
    z_norm = np.clip((pts[:,2] + 0.3) / 1.5, 0, 1)

    gt_pt_counts = [count_points_in_box(all_points, b) for b in gt_boxes]
    xlim = (cx-margin, cx+margin)
    ylim = (cy-margin, cy+margin)
    zlim = (-0.3, 1.2)

    fig = plt.figure(figsize=(27, 10), facecolor='#0d0d1a')
    fig.suptitle(
        f'VoxelNeXt Cone Detection  —  Frame {FRAME_IDX}  (ID: {lidar_idx})\n'
        f'Gesamt: {total_pts:,} Punkte  |  Zoom: {zoom_pts:,} Punkte  '
        f'|  {len(gt_boxes)} Cones  (max. {max_dist:.1f} m)',
        fontsize=13, fontweight='bold', color='white', y=1.02)

    ax_raw = fig.add_subplot(1, 3, 1, projection='3d')
    ax_bev = fig.add_subplot(1, 3, 2)
    ax_3d  = fig.add_subplot(1, 3, 3, projection='3d')

    # ── ROHE PUNKTWOLKE ──────────────────────────────────────────
    ax_raw.scatter(pts[:,0], pts[:,1], pts[:,2],
                   s=4, c=z_norm, cmap=cone_cmap,
                   alpha=0.95, vmin=0, vmax=1)
    setup_3d_ax(ax_raw, xlim, ylim, zlim, azim=azim, elev=20)
    ax_raw.set_title(f'Rohe LiDAR-Punktwolke\n{zoom_pts:,} Punkte',
                     fontsize=11, fontweight='bold', color='white', pad=8)
    # Legende Boden/Cone
    from matplotlib.lines import Line2D
    legend_raw = [
        Line2D([0],[0], marker='o', color='w', markerfacecolor='#2c2c3e',
               markersize=8, label='Boden (Z < 0.1 m)'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor='#f0e040',
               markersize=8, label='Cone-Basis'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor='#ff2200',
               markersize=8, label='Cone-Spitze'),
    ]
    ax_raw.legend(handles=legend_raw, loc='upper left', fontsize=8,
                  framealpha=0.85, facecolor='#0a0a1a', labelcolor='white',
                  edgecolor='white')

    # ── BEV ──────────────────────────────────────────────────────
    ax_bev.set_facecolor('#1a1a2e')
    dist_bev = np.sqrt(pts[:,0]**2 + pts[:,1]**2)
    ax_bev.scatter(pts[:,0], pts[:,1], s=2, c=dist_bev, cmap='plasma',
                   alpha=0.8, vmin=0, vmax=max_dist+2)

    for i, box in enumerate(gt_boxes):
        is_fn = i in fn_indices
        color = COLORS['FN'] if is_fn else COLORS['GT']
        draw_bev_box(ax_bev, box, color, linestyle='--' if is_fn else '-',
                     label='FN' if is_fn else 'GT')
        ax_bev.text(box[0], box[1]-0.4, f'{gt_pt_counts[i]}p',
                    color=color, fontsize=7, ha='center', va='top',
                    bbox=dict(boxstyle='round,pad=0.1', facecolor='#1a1a2e',
                              alpha=0.8, edgecolor='none'))

    for i, box in enumerate(pred_boxes):
        draw_bev_box(ax_bev, box, COLORS[pred_labels[i]],
                     label=pred_labels[i], score=pred_scores[i])

    ax_bev.set_xlim(*xlim); ax_bev.set_ylim(*ylim)
    ax_bev.set_xlabel('X (m)', color='white', fontsize=10)
    ax_bev.set_ylabel('Y (m)', color='white', fontsize=10)
    ax_bev.tick_params(colors='white')
    ax_bev.set_title(f"Bird's Eye View  |  TP={tp}  FP={fp}  FN={fn}",
                     color='white', fontsize=11, fontweight='bold', pad=8)
    ax_bev.set_aspect('equal')
    ax_bev.grid(True, alpha=0.2, color='white')
    ax_bev.legend(handles=[
        mpatches.Patch(color=COLORS['GT'], label=f'GT ({len(gt_boxes)})'),
        mpatches.Patch(color=COLORS['TP'], label=f'TP ({tp})'),
        mpatches.Patch(color=COLORS['FP'], label=f'FP ({fp})'),
        mpatches.Patch(color=COLORS['FN'], label=f'FN ({fn})'),
    ], loc='upper right', fontsize=9, framealpha=0.85,
       facecolor='#1a1a2e', labelcolor='white', edgecolor='white')

    # ── 3D MIT BOXEN ─────────────────────────────────────────────
    ax_3d.scatter(pts[:,0], pts[:,1], pts[:,2],
                  s=4, c=z_norm, cmap=cone_cmap, alpha=0.9, vmin=0, vmax=1)

    for i, box in enumerate(gt_boxes):
        col  = COLORS['FN'] if i in fn_indices else COLORS['GT']
        cors = get_corners(box)
        segs = [[cors[a], cors[b]] for a,b in LINES]
        ax_3d.add_collection3d(Line3DCollection(segs, colors=col, linewidths=3))
        ax_3d.text(box[0], box[1], box[2]+box[5]/2+0.08,
                   f'{gt_pt_counts[i]}p', color=col, fontsize=7,
                   ha='center', fontweight='bold')

    for i, box in enumerate(pred_boxes):
        cors = get_corners(box)
        segs = [[cors[a], cors[b]] for a,b in LINES]
        ax_3d.add_collection3d(Line3DCollection(segs, colors=COLORS[pred_labels[i]], linewidths=3))

    setup_3d_ax(ax_3d, xlim, ylim, zlim, azim=azim, elev=20)
    ax_3d.set_title('3D LiDAR + Detektion',
                    fontsize=11, fontweight='bold', color='white', pad=8)

    plt.tight_layout()
    out = f"/mnt/c/Users/Student/Desktop/VoxelNext_V2/pub_frame_{FRAME_IDX:02d}_{lidar_idx}.png"
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0d0d1a')
    plt.close()
    print(f"✅ Frame {FRAME_IDX:02d} | TP={tp} FP={fp} FN={fn} | azim={azim:.0f}°")

print("\n🎉 Fertig!")
