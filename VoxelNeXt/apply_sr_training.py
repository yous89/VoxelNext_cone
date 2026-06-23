
import numpy as np
import pickle
import torch
import torch.nn as nn
import shutil, os

DATA_ROOT = "/mnt/c/Users/Student/Desktop/VoxelNext_V2/data/kitti_cones"

class ConePointSR(nn.Module):
    def __init__(self, n_input=4, n_output=20):
        super().__init__()
        self.n_input  = n_input
        self.n_output = n_output
        self.encoder  = nn.Sequential(
            nn.Linear(3, 32),   nn.ReLU(),
            nn.Linear(32, 64),  nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(128, 256),          nn.ReLU(),
            nn.Linear(256, 512),          nn.ReLU(),
            nn.Linear(512, n_output * 3), nn.Tanh(),
        )
    def forward(self, x):
        B     = x.shape[0]
        feat  = self.encoder(x)
        gfeat = feat.max(dim=1)[0]
        out   = self.decoder(gfeat)
        out   = out.view(B, self.n_output, 3)
        return out * 0.15

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = ConePointSR(n_input=4, n_output=20).to(device)
model.load_state_dict(torch.load("cone_sr_model.pth", map_location=device))
model.eval()
print(f"SR-Modell geladen! Device: {device}")

def super_resolve_cone(sparse_pts, center):
    pts_norm = sparse_pts[:, :3] - center
    if len(pts_norm) < 4:
        idx = np.random.choice(len(pts_norm), 4, replace=True)
    else:
        idx = np.random.choice(len(pts_norm), 4, replace=False)
    pts_norm = pts_norm[idx]
    inp  = torch.FloatTensor(pts_norm).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(inp).squeeze(0).cpu().numpy()
    dense_pts = out + center
    intens    = np.full((len(dense_pts), 1),
                        sparse_pts[:,3].mean() if sparse_pts.shape[1]==4 else 5.0)
    return np.hstack([dense_pts, intens])

def apply_sr_to_split(infos, split_name, backup_suffix):
    BACKUP_DIR = DATA_ROOT + "/training/velodyne_" + backup_suffix
    os.makedirs(BACKUP_DIR, exist_ok=True)

    total_sr  = 0
    total_pts = 0

    print(f"Wende SR auf {split_name} an ({len(infos)} Frames)...")
    for i, gt_info in enumerate(infos):
        lidar_idx = gt_info["point_cloud"]["lidar_idx"]
        src = DATA_ROOT + "/training/velodyne/" + lidar_idx + ".bin"

        if not os.path.exists(src):
            continue

        pw = np.fromfile(src, dtype=np.float32).reshape(-1, 4)
        gt_boxes = gt_info["annos"]["gt_boxes_lidar"]
        gt_names = gt_info["annos"]["name"]
        gt_boxes = gt_boxes[gt_names == "small_cone"]

        neue_punkte = []
        n_sr = 0

        for box in gt_boxes:
            x,y,z,dx,dy,dz,_ = box
            dist = np.sqrt(x**2 + y**2)
            if not (10 <= dist < 15):
                continue
            in_box = ((pw[:,0]>=x-dx/2)&(pw[:,0]<=x+dx/2)&
                      (pw[:,1]>=y-dy/2)&(pw[:,1]<=y+dy/2)&
                      (pw[:,2]>=z-dz/2)&(pw[:,2]<=z+dz/2))
            cone_pts = pw[in_box]
            if len(cone_pts) < 1 or len(cone_pts) >= 10:
                continue
            center    = np.array([x, y, z])
            dense_pts = super_resolve_cone(cone_pts, center)
            neue_punkte.append(dense_pts)
            n_sr += 1
            total_pts += len(dense_pts)

        # Backup + speichern
        if not os.path.exists(BACKUP_DIR + "/" + lidar_idx + ".bin"):
            shutil.copy(src, BACKUP_DIR + "/" + lidar_idx + ".bin")

        if neue_punkte:
            pw_neu = np.vstack([pw] + neue_punkte).astype(np.float32)
            pw_neu.tofile(src)

        total_sr += n_sr
        if i % 20 == 0:
            print(f"  Frame {i:3d}/{len(infos)} | {n_sr} Cones SR")

    print(f"  Gesamt: {total_sr} Cones, {total_pts} neue Punkte")
    return total_sr

# Trainingsdaten
with open(DATA_ROOT + "/kitti_infos_train.pkl", "rb") as f:
    train_infos = pickle.load(f)

n_train = apply_sr_to_split(train_infos, "TRAINING", "train_backup_sr")
print()
print(f"Fertig! {n_train} Training-Cones super-resolved")
print()
print("Jetzt neu trainieren:")
print("  cd tools")
print("  python train.py --cfg_file cfgs/kitti_models/voxelnext_cones.yaml --batch_size 4")
