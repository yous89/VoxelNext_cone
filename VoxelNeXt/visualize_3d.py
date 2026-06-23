import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')  # kein Display nötig
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

with open("output/kitti_models/voxelnext_cones/default/eval/epoch_50/val/default/result.pkl", 'rb') as f:
    predictions = pickle.load(f)
with open("/mnt/c/Users/Student/Desktop/VoxelNext_V2/data/kitti_cones/kitti_infos_val.pkl", 'rb') as f:
    gt_infos = pickle.load(f)

def get_box_corners(box):
    x, y, z, dx, dy, dz, rot = box
    corners = np.array([
        [ dx/2,  dy/2, -dz/2], [-dx/2,  dy/2, -dz/2],
        [-dx/2, -dy/2, -dz/2], [ dx/2, -dy/2, -dz/2],
        [ dx/2,  dy/2,  dz/2], [-dx/2,  dy/2,  dz/2],
        [-dx/2, -dy/2,  dz/2], [ dx/2, -dy/2,  dz/2],
    ])
    c, s = np.cos(rot), np.sin(rot)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return (R @ corners.T).T + np.array([x, y, z])

def box_lines(corners):
    idx = [[0,1],[1,2],[2,3],[3,0],
           [4,5],[5,6],[6,7],[7,4],
           [0,4],[1,5],[2,6],[3,7]]
    return [[corners[i], corners[j]] for i,j in idx]

for FRAME_IDX in range(min(6, len(predictions))):
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

    fig = plt.figure(figsize=(16, 8))

    # === BEV (Bird Eye View) ===
    ax1 = fig.add_subplot(121)
    # Punkte
    ax1.scatter(points[:, 0], points[:, 1], s=0.3, c='gray', alpha=0.4)
    # GT Boxen
    for box in gt_boxes:
        c = get_box_corners(box)
        bev = c[[0,1,2,3,0], :2]
        ax1.plot(bev[:,0], bev[:,1], 'g-', linewidth=1.5, label='GT')
    # Pred Boxen
    for i, box in enumerate(pred_boxes):
        c = get_box_corners(box)
        bev = c[[0,1,2,3,0], :2]
        ax1.plot(bev[:,0], bev[:,1], 'r-', linewidth=1.5, label='Pred')
        ax1.text(box[0], box[1], f'{pred_scores[i]:.2f}', color='red', fontsize=6)
    ax1.set_xlim(-35, 35); ax1.set_ylim(-35, 35)
    ax1.set_xlabel('X (m)'); ax1.set_ylabel('Y (m)')
    ax1.set_title(f'BEV - Frame {FRAME_IDX} ({lidar_idx})\nGT={len(gt_boxes)} (grün)  Pred={len(pred_boxes)} (rot)')
    ax1.set_aspect('equal'); ax1.grid(True, alpha=0.3)
    handles = [plt.Line2D([0],[0],color='g',label=f'GT ({len(gt_boxes)})'),
               plt.Line2D([0],[0],color='r',label=f'Pred ({len(pred_boxes)})')]
    ax1.legend(handles=handles)

    # === 3D View ===
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(points[::5, 0], points[::5, 1], points[::5, 2],
                s=0.2, c='gray', alpha=0.3)
    for box in gt_boxes:
        segs = box_lines(get_box_corners(box))
        ax2.add_collection3d(Line3DCollection(segs, colors='green', linewidths=1.5))
    for box in pred_boxes:
        segs = box_lines(get_box_corners(box))
        ax2.add_collection3d(Line3DCollection(segs, colors='red', linewidths=1.5))
    ax2.set_xlim(-15, 15); ax2.set_ylim(-15, 15); ax2.set_zlim(-2, 3)
    ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('Z')
    ax2.set_title('3D View')
    ax2.view_init(elev=25, azim=-60)

    plt.tight_layout()
    out = f"/mnt/c/Users/Student/Desktop/VoxelNext_V2/viz_frame_{FRAME_IDX:02d}_{lidar_idx}.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Gespeichert: {out}")

print("\n✅ Alle Frames gespeichert auf dem Windows Desktop!")
