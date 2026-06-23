
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

DATA_ROOT = "/mnt/c/Users/Student/Desktop/VoxelNext_V2/data/kitti_cones"

CONE_HEIGHT = 0.325
CONE_RADIUS = 0.114
N_INPUT     = 4    # Eingabe-Punkte (LOW-RES)
N_OUTPUT    = 20   # Ausgabe-Punkte (HIGH-RES)

# ═══════════════════════════════════════════════════
# 1. DATEN LADEN
# ═══════════════════════════════════════════════════
def load_cone_points(infos, min_pts=10, max_dist=5.0):
    cones = []
    for gt_info in infos:
        lidar_idx = gt_info["point_cloud"]["lidar_idx"]
        pw = np.fromfile(DATA_ROOT + "/training/velodyne/" + lidar_idx + ".bin",
                         dtype=np.float32).reshape(-1, 4)
        gt_boxes = gt_info["annos"]["gt_boxes_lidar"]
        gt_names = gt_info["annos"]["name"]
        gt_boxes = gt_boxes[gt_names == "small_cone"]
        for box in gt_boxes:
            x,y,z,dx,dy,dz,_ = box
            dist = np.sqrt(x**2 + y**2)
            if dist > max_dist: continue
            in_box = ((pw[:,0]>=x-dx/2)&(pw[:,0]<=x+dx/2)&
                      (pw[:,1]>=y-dy/2)&(pw[:,1]<=y+dy/2)&
                      (pw[:,2]>=z-dz/2)&(pw[:,2]<=z+dz/2))
            pts = pw[in_box]
            if len(pts) >= min_pts:
                # Normalisiere auf Cone-Zentrum
                center = np.array([x, y, z])
                pts_norm = pts[:, :3] - center
                cones.append({"pts": pts_norm, "center": center, "box": box})
    return cones

# ═══════════════════════════════════════════════════
# 2. DATASET
# ═══════════════════════════════════════════════════
class ConeDataset(Dataset):
    def __init__(self, cones, n_input=4, n_output=20, augment=True):
        self.cones    = cones
        self.n_input  = n_input
        self.n_output = n_output
        self.augment  = augment

    def __len__(self):
        return len(self.cones) * 10  # Augmentierung

    def __getitem__(self, idx):
        cone = self.cones[idx % len(self.cones)]
        pts  = cone["pts"]

        # HIGH-RES: zufällig n_output Punkte samplen
        if len(pts) >= self.n_output:
            idx_hr = np.random.choice(len(pts), self.n_output, replace=False)
        else:
            idx_hr = np.random.choice(len(pts), self.n_output, replace=True)
        high_res = pts[idx_hr]

        # LOW-RES: zufällig n_input Punkte aus high_res
        idx_lr = np.random.choice(self.n_output, self.n_input, replace=False)
        low_res = high_res[idx_lr]

        # Augmentierung: zufällige Rotation um Z-Achse
        if self.augment:
            angle = np.random.uniform(0, 2*np.pi)
            c, s  = np.cos(angle), np.sin(angle)
            R     = np.array([[c,-s,0],[s,c,0],[0,0,1]])
            low_res  = (R @ low_res.T).T
            high_res = (R @ high_res.T).T

        return (torch.FloatTensor(low_res),
                torch.FloatTensor(high_res))

# ═══════════════════════════════════════════════════
# 3. MODELL: Leichtgewichtiges PointNet SR
# ═══════════════════════════════════════════════════
class ConePointSR(nn.Module):
    def __init__(self, n_input=4, n_output=20):
        super().__init__()
        self.n_input  = n_input
        self.n_output = n_output

        # Encoder: verarbeite jeden Eingabepunkt
        self.encoder = nn.Sequential(
            nn.Linear(3, 32),   nn.ReLU(),
            nn.Linear(32, 64),  nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
        )

        # Global Feature durch Max-Pooling
        # Input: n_input * 128 → Global: 128

        # Decoder: generiere n_output Punkte
        self.decoder = nn.Sequential(
            nn.Linear(128, 256),          nn.ReLU(),
            nn.Linear(256, 512),          nn.ReLU(),
            nn.Linear(512, n_output * 3), nn.Tanh(),
        )

    def forward(self, x):
        # x: (B, n_input, 3)
        B = x.shape[0]

        # Encode jeden Punkt
        feat = self.encoder(x)          # (B, n_input, 128)

        # Global Max Pooling
        global_feat = feat.max(dim=1)[0]  # (B, 128)

        # Decode zu n_output Punkten
        out = self.decoder(global_feat)   # (B, n_output*3)
        out = out.view(B, self.n_output, 3)

        # Skaliere auf Cone-Größe
        out = out * 0.15  # Cone Radius ~0.114m

        return out

# ═══════════════════════════════════════════════════
# 4. VERLUSTFUNKTION: Chamfer Distance
# ═══════════════════════════════════════════════════
def chamfer_distance(pred, target):
    # pred:   (B, N, 3)
    # target: (B, M, 3)
    pred_exp   = pred.unsqueeze(2)    # (B, N, 1, 3)
    target_exp = target.unsqueeze(1)  # (B, 1, M, 3)
    dist = ((pred_exp - target_exp)**2).sum(-1)  # (B, N, M)

    # Für jeden Pred-Punkt: nächster Target-Punkt
    loss1 = dist.min(dim=2)[0].mean()
    # Für jeden Target-Punkt: nächster Pred-Punkt
    loss2 = dist.min(dim=1)[0].mean()

    return loss1 + loss2

# ═══════════════════════════════════════════════════
# 5. TRAINING
# ═══════════════════════════════════════════════════
def train():
    with open(DATA_ROOT + "/kitti_infos_train.pkl", "rb") as f:
        train_infos = pickle.load(f)
    with open(DATA_ROOT + "/kitti_infos_val.pkl", "rb") as f:
        val_infos = pickle.load(f)

    print("Lade Cone-Daten...")
    train_cones = load_cone_points(train_infos, min_pts=10, max_dist=5.0)
    val_cones   = load_cone_points(val_infos,   min_pts=10, max_dist=5.0)
    print(f"  Training: {len(train_cones)} Cones")
    print(f"  Val:      {len(val_cones)} Cones")

    if len(train_cones) < 10:
        print("Zu wenig Daten! Versuche mit max_dist=8m")
        train_cones = load_cone_points(train_infos, min_pts=8, max_dist=8.0)
        val_cones   = load_cone_points(val_infos,   min_pts=8, max_dist=8.0)
        print(f"  Training: {len(train_cones)} Cones")

    train_ds = ConeDataset(train_cones, n_input=N_INPUT, n_output=N_OUTPUT)
    val_ds   = ConeDataset(val_cones,   n_input=N_INPUT, n_output=N_OUTPUT, augment=False)

    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model     = ConePointSR(n_input=N_INPUT, n_output=N_OUTPUT).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    print()
    print(f"{'Epoch':>6} | {'Train Loss':>11} | {'Val Loss':>10} | {'LR':>8}")
    print("-"*45)

    best_val_loss = float("inf")
    for epoch in range(1, 101):
        # Training
        model.train()
        train_loss = 0
        for low_res, high_res in train_dl:
            low_res  = low_res.to(device)
            high_res = high_res.to(device)
            pred     = model(low_res)
            loss     = chamfer_distance(pred, high_res)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_dl)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for low_res, high_res in val_dl:
                low_res  = low_res.to(device)
                high_res = high_res.to(device)
                pred     = model(low_res)
                loss     = chamfer_distance(pred, high_res)
                val_loss += loss.item()
        val_loss /= len(val_dl)
        scheduler.step()

        if epoch % 10 == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(f"{epoch:>6} | {train_loss:>11.6f} | {val_loss:>10.6f} | {lr:>8.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "cone_sr_model.pth")

    print()
    print(f"Bestes Modell gespeichert: cone_sr_model.pth")
    print(f"Beste Val Loss: {best_val_loss:.6f}")
    return model

if __name__ == "__main__":
    train()
