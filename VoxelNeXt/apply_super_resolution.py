
import numpy as np
import pickle
import torch
import torch.nn as nn
import shutil, os

DATA_ROOT = "/mnt/c/Users/Student/Desktop/VoxelNext_V2/data/kitti_cones"

# ═══════════════════════════════════════════
# Modell laden
# ═══════════════════════════════════════════
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
        B        = x.shape[0]
        feat     = self.encoder(x)
        gfeat    = feat.max(dim=1)[0]
        out      = self.decoder(gfeat)
        out      = out.view(B, self.n_output, 3)
        return out * 0.15

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = ConePointSR(n_input=4, n_output=20).to(device)
model.load_state_dict(torch.load("cone_sr_model.pth", map_location=device))
model.eval()
print(f"Modell geladen! Device: {device}")

# ═══════════════════════════════════════════
# Super Resolution Funktion
# ═══════════════════════════════════════════
def super_resolve_cone(sparse_pts, center):
    pts_norm = sparse_pts[:, :3] - center

    # Padding falls zu wenig Punkte
    if len(pts_norm) < 4:
        idx = np.random.choice(len(pts_norm), 4, replace=True)
        pts_norm = pts_norm[idx]
    else:
        idx = np.random.choice(len(pts_norm), 4, replace=False)
        pts_norm = pts_norm[idx]

    inp  = torch.FloatTensor(pts_norm).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(inp).squeeze(0).cpu().numpy()

    # Zurück in Weltkoordinaten
    dense_pts = out + center
    intens    = np.full((len(dense_pts), 1),
                        sparse_pts[:,3].mean() if sparse_pts.shape[1]==4 else 5.0)
    return np.hstack([dense_pts, intens])

# ═══════════════════════════════════════════
# Anwenden auf Val-Punktwolken
# GT-Box bekannt → gezielt anwenden
# ═══════════════════════════════════════════
with open(DATA_ROOT + "/kitti_infos_val.pkl", "rb") as f:
    gt_infos = pickle.load(f)

BACKUP_DIR = DATA_ROOT + "/training/velodyne_backup_sr"
INTERP_DIR = DATA_ROOT + "/training/velodyne_sr"
os.makedirs(BACKUP_DIR, exist_ok=True)
os.makedirs(INTERP_DIR, exist_ok=True)

print("Wende Super Resolution auf Val-Punktwolken an...")
print("(Nur Cones bei 10-15m mit 2-8 Punkten)")
print()

total_sr   = 0
total_pts_added = 0

for i, gt_info in enumerate(gt_infos):
    lidar_idx = gt_info["point_cloud"]["lidar_idx"]
    src = DATA_ROOT + "/training/velodyne/" + lidar_idx + ".bin"
    pw  = np.fromfile(src, dtype=np.float32).reshape(-1, 4)

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

        # Punkte in Box finden
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
        total_pts_added += len(dense_pts)

    if neue_punkte:
        pw_neu = np.vstack([pw] + neue_punkte).astype(np.float32)
    else:
        pw_neu = pw

    # Backup + speichern
    shutil.copy(src, BACKUP_DIR + "/" + lidar_idx + ".bin")
    pw_neu.tofile(INTERP_DIR + "/" + lidar_idx + ".bin")

    total_sr += n_sr
    print(f"  Frame {i:2d} | {lidar_idx[-15:]:>20} | "
          f"{n_sr} Cones SR | "
          f"{len(pw)} -> {len(pw_neu)} Punkte")

print()
print(f"Gesamt: {total_sr} Cones super-resolved")
print(f"Gesamt: {total_pts_added} neue Punkte hinzugefügt")

# Ersetze originale mit SR-Version
print()
print("Ersetze Punktwolken...")
for gt_info in gt_infos:
    lidar_idx = gt_info["point_cloud"]["lidar_idx"]
    shutil.copy(INTERP_DIR + "/" + lidar_idx + ".bin",
                DATA_ROOT + "/training/velodyne/" + lidar_idx + ".bin")

print("Fertig! Starte VoxelNeXt Evaluation:")
print()
print("  cd tools")
print("  python test.py --cfg_file cfgs/kitti_models/voxelnext_cones.yaml")
print("      --ckpt ../output/kitti_models/voxelnext_cones/default/ckpt/checkpoint_epoch_50.pth")
print("      --batch_size 4 --eval_tag with_super_resolution")
