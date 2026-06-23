import os, shutil, pickle

DATA_ROOT  = "/mnt/c/Users/Student/Desktop/VoxelNext_V2/data/kitti_cones"
BACKUP_DIR = DATA_ROOT + "/training/velodyne_backup"

with open(DATA_ROOT + "/kitti_infos_val.pkl", "rb") as f:
    gt_infos = pickle.load(f)

print("Stelle originale Punktwolken wieder her...")
for gt_info in gt_infos:
    lidar_idx = gt_info["point_cloud"]["lidar_idx"]
    src = BACKUP_DIR + "/" + lidar_idx + ".bin"
    dst = DATA_ROOT + "/training/velodyne/" + lidar_idx + ".bin"
    if os.path.exists(src):
        shutil.copy(src, dst)
print("Originale wiederhergestellt!")
