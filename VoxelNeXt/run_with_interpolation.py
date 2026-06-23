import numpy as np
import pickle
import os
import shutil
from sklearn.cluster import DBSCAN

DATA_ROOT = "/mnt/c/Users/Student/Desktop/VoxelNext_V2/data/kitti_cones"

CONE_HEIGHT = 0.325
CONE_RADIUS = 0.114

def kegel_interpolation(sparse_points, ziel_punkte=20):
    pts     = sparse_points[:, :3]
    zentrum = pts.mean(axis=0)
    basis_z = pts[:, 2].min()
    n_neu   = max(0, ziel_punkte - len(pts))
    neue    = []
    for _ in range(n_neu):
        h      = np.random.uniform(0, CONE_HEIGHT)
        r      = CONE_RADIUS * (1 - h / CONE_HEIGHT)
        winkel = np.random.uniform(0, 2 * np.pi)
        neue.append([zentrum[0] + r*np.cos(winkel),
                     zentrum[1] + r*np.sin(winkel),
                     basis_z + h])
    if neue:
        return np.vstack([pts, np.array(neue)])
    return pts

def upsampling(punktwolke, min_dist=10.0, max_dist=15.0,
               min_punkte=2, max_punkte=8, ziel_punkte=20):
    dist  = np.sqrt(punktwolke[:,0]**2 + punktwolke[:,1]**2)
    maske = (dist >= min_dist) & (dist <= max_dist)
    fern  = punktwolke[maske]
    nah   = punktwolke[~maske]
    if len(fern) < min_punkte:
        return punktwolke, 0
    labels  = DBSCAN(eps=0.3, min_samples=min_punkte).fit_predict(fern[:,:3])
    interp  = []
    n_found = 0
    for lbl in set(labels):
        if lbl == -1:
            continue
        cluster = fern[labels == lbl]
        if min_punkte <= len(cluster) <= max_punkte:
            dichte   = kegel_interpolation(cluster, ziel_punkte)
            intens   = np.full((len(dichte)-len(cluster), 1), cluster[:,3].mean())
            neue_pts = np.hstack([dichte[len(cluster):], intens])
            interp.append(np.vstack([cluster, neue_pts]))
            n_found += 1
    if interp:
        return np.vstack([nah, fern, *interp]), n_found
    return punktwolke, 0

with open(DATA_ROOT + "/kitti_infos_val.pkl", "rb") as f:
    gt_infos = pickle.load(f)

BACKUP_DIR = DATA_ROOT + "/training/velodyne_backup"
INTERP_DIR = DATA_ROOT + "/training/velodyne_interp"
os.makedirs(BACKUP_DIR, exist_ok=True)
os.makedirs(INTERP_DIR, exist_ok=True)

print("Erstelle interpolierte Punktwolken...")
total_found = 0
for i, gt_info in enumerate(gt_infos):
    lidar_idx = gt_info["point_cloud"]["lidar_idx"]
    src = DATA_ROOT + "/training/velodyne/" + lidar_idx + ".bin"
    pw  = np.fromfile(src, dtype=np.float32).reshape(-1, 4)
    pw_neu, n = upsampling(pw)
    total_found += n
    pw_neu.astype(np.float32).tofile(INTERP_DIR + "/" + lidar_idx + ".bin")
    shutil.copy(src, BACKUP_DIR + "/" + lidar_idx + ".bin")
    print("  Frame " + str(i) + " | " + lidar_idx + " | " + str(n) + " Cones gefunden | " + str(len(pw)) + " -> " + str(len(pw_neu)) + " Punkte")

print("Gesamt: " + str(total_found) + " Cone-Kandidaten interpoliert")
print("Ersetze Punktwolken...")
for gt_info in gt_infos:
    lidar_idx = gt_info["point_cloud"]["lidar_idx"]
    shutil.copy(INTERP_DIR + "/" + lidar_idx + ".bin",
                DATA_ROOT + "/training/velodyne/" + lidar_idx + ".bin")
print("Fertig! Jetzt VoxelNeXt evaluation starten:")
print("")
print("  cd tools")
print("  python test.py --cfg_file cfgs/kitti_models/voxelnext_cones.yaml --ckpt ../output/kitti_models/voxelnext_cones/default/ckpt/checkpoint_epoch_50.pth --batch_size 4 --eval_tag with_interpolation")
