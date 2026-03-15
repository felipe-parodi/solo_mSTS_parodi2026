# visual features to control for in figure 3

TL;DR answers

“Virtual head view” = take your fixed scene frames and warp them to a tiny ground-plane patch placed ~1 m ahead of the head, oriented by the 3D head pose. That patch then changes exactly as the head translates/rotates → you recover the optic-flow geometry your retina would see (expansion/divergence for forward motion, curl for yaw).

Use 3D head pose? Yes—that’s the point. Head position picks the patch center; head orientation picks the patch axes; you then project through camera intrinsics/extrinsics to sample pixels.

Geometry cheat-sheet (world → camera)

Assume you store intrinsics 
𝐾
K and extrinsics 
𝑅
,
𝑡
R,t so:

𝑋
𝑐
=
𝑅
 
𝑋
𝑤
+
𝑡
,
𝑝
∼
𝐾
 
𝑋
𝑐
X
c
	​

=RX
w
	​

+t,p∼KX
c
	​


Camera forward in world: 
𝑓
^
cam
=
𝑅
⊤
[
0
,
0
,
1
]
⊤
f
^
	​

cam
	​

=R
⊤
[0,0,1]
⊤
.

Visibility weight: 
𝑤
=
max
⁡
{
0
,
 
𝑓
^
head
⋅
𝑓
^
cam
}
w=max{0, 
f
^
	​

head
	​

⋅
f
^
	​

cam
	​

}.

Ray–plane intersection (point ~L ahead, but on the plane):

𝑄
𝑤
=
𝑃
head
+
𝑠
 
𝑓
^
head
,
𝑠
=
−
𝑛
⊤
𝑃
head
+
𝑑
𝑛
⊤
𝑓
^
head
.
Q
w
	​

=P
head
	​

+s
f
^
	​

head
	​

,s=−
n
⊤
f
^
	​

head
	​

n
⊤
P
head
	​

+d
	​

.

Plane homography (world plane → image):

𝐻
=
𝐾
 
(
𝑅
−
𝑡
 
𝑛
⊤
/
𝑑
)
(
for points on 
Π
)
.
H=K(R−tn
⊤
/d)(for points on Π).

(If signs seem off, test with a few labeled floor points; flip 
𝑛
,
𝑑
n,d together if needed.)

“Virtual head view” in one camera (concise pseudo-code)

Single camera first; everything vectorized where it matters.

# Inputs per bin: P_head (3,), f_head_unit (3,), frames at t-1 and t
# Camera calib: K (3x3), R (3x3), t (3,), plane: n (3,), d (scalar)

# 1) Choose center on the floor ahead of head
s = -(n @ P_head + d) / (n @ f_head_unit)
Qw = P_head + s * f_head_unit   # world point where the head-forward ray meets the floor

# 2) Build a head-aligned tangent basis on the plane
f_proj = f_head_unit - (n @ f_head_unit) * n / (np.linalg.norm(n)**2)
e1 = f_proj / np.linalg.norm(f_proj)           # "forward" on the floor
e2 = np.cross(n, e1); e2 /= np.linalg.norm(e2) # "left/right" on the floor

# 3) Define a small canonical patch around Qw (meters-per-pixel = mpp)
H_px, W_px = 200, 200
cx, cy = (W_px-1)/2.0, (H_px-1)/2.0
# Build world coords for all patch pixels (vectorized):
# Xw(i,j) = Qw + ((j-cx)*mpp)*e1 + ((i-cy)*mpp)*e2
J, I = np.meshgrid(np.arange(W_px), np.arange(H_px))  # J ~ x/cols, I ~ y/rows
Xw = (Qw[:,None,None]
      + ((J - cx)*mpp)[None,:,:]*e1[:,None,None]
      + ((I - cy)*mpp)[None,:,:]*e2[:,None,None])          # shape (3,H,W)

# 4) Project into the camera to sample pixels (inverse mapping)
Xc = R @ Xw.reshape(3,-1) + t[:,None]                      # (3, H*W)
u = K[0,0]*Xc[0]/Xc[2] + K[0,2]
v = K[1,1]*Xc[1]/Xc[2] + K[1,2]
# Bilinear sample frame_{t-1} and frame_t at (u,v) → patch_prev, patch_curr (H,W)

# 5) Dense flow on patches (Farnebäck or TV-L1)
u_flow, v_flow = dense_flow(patch_prev, patch_curr)

# 6) ROI = disk around center; compute stats
mask = disk(center=(cy,cx), radius=ROI_RADIUS_PX, shape=(H_px,W_px))
mag = np.hypot(u_flow, v_flow)
mag_mean = mag[mask].mean()

du_dx, du_dy = np.gradient(u_flow)
dv_dx, dv_dy = np.gradient(v_flow)
div  = (du_dx + dv_dy)[mask].mean()
curl = (dv_dx - du_dy)[mask].mean()

vx = u_flow[mask].mean(); vy = v_flow[mask].mean()
norm = np.hypot(vx, vy) + 1e-8
dir_x, dir_y = vx/norm, vy/norm

# Save per-bin: [mag_mean, div, curl, dir_x, dir_y, w]


Why this works here: your arena is static; warping to the head-centric ground patch makes the pixels move because the head moved, recovering the retinal slip cues (expansion for forward motion, curl for yaw) without a headcam.

Motion-Energy (ME) PCs on the same patch

Use frame-difference (or a tiny temporal filter) + a bank of 2D Gabors on the patch (inside the same ROI). Square, average within ROI → energy vector → PCA (fit once per session on valid bins). Insert PC scores per bin (NaN where invalid), z-score per session. This adds direction/speed-selective texture responses beyond plain flow.

1) Design doc (for a coding LLM)
Problem

We want visual-only covariates approximating the retinal consequences of self-motion (expansion/divergence, rotation/curl, magnitude, dominant direction, and motion-energy PCs) to use in encoding/variance partitioning for mid-STS. The dataset has:

Static arena, static scene cameras (multi-camera system).

Only the monkey moves (no background motion).

We have 3D head pose (position + orientation), camera intrinsics/extrinsics, and a calibrated floor plane (floor_plane.json with plane 
𝑎
,
𝑏
,
𝑐
,
𝑑
a,b,c,d in world).

Why the original plan (raw optic flow at a head-point ROI in each scene camera) fails here

With fixed cameras and a static arena, background pixels don’t move when the monkey moves; only the monkey’s body moves within the frame.

Computing optical flow inside a raw ROI centered where the head is “looking” in a scene camera mostly yields ~zero flow unless the monkey’s body/shadow crosses that ROI. That does not approximate the retinal slip caused by self-translation/rotation.

Consequence: VIS regressors are artificially weak → you might wrongly attribute variance to KIN simply because VIS didn’t capture the right signal.

Solution: head-centric virtual view via planar warp

Use the floor plane Π: 
𝑛
⊤
𝑋
+
𝑑
=
0
n
⊤
X+d=0 in world coords and the 3D head pose.

For each bin, cast a ray from head position along head-forward to intersect Π at 
𝑄
𝑤
Q
w
	​

.

Around 
𝑄
𝑤
Q
w
	​

, build a small tangent patch on Π in head-aligned axes (forward = projected head-forward; lateral = cross with plane normal).

Warp each scene camera’s frame into this canonical virtual patch (homography sampling), both at 
𝑡
−
1
t−1 and 
𝑡
t.

Now the patch changes because the head moved, recovering expansion (divergence) with forward speed and curl with yaw rate.

Compute dense flow on the virtual patch; summarize in a circular ROI (magnitude, divergence, curl, dominant direction). Add motion-energy features (small Gabor bank on frame difference) → PCA → ME PCs.

Visibility weight per camera: 
𝑤
=
max
⁡
(
0
,
cos
⁡
(
𝑓
^
head
,
𝑓
^
cam
)
)
w=max(0,cos(
f
^
	​

head
	​

,
f
^
	​

cam
	​

)), where 
𝑓
^
cam
=
𝑅
⊤
[
0
,
0
,
1
]
f
^
	​

cam
	​

=R
⊤
[0,0,1] if 
𝑋
𝑐
=
𝑅
𝑋
𝑤
+
𝑡
X
c
	​

=RX
w
	​

+t.

Aggregate features across cameras with these weights; keep wsum & cams_used.

Outputs

Per 50 ms bin (no look-ahead):
flow_mag, flow_div, flow_curl, flow_dir_x, flow_dir_y, me_pc1..K, wsum, cams_used → vis_features.parquet (z-scored per session).

Validation expectations

corr
(
divergence
,
forward speed
)
≫
0
corr(divergence,forward speed)≫0

corr
(
curl
,
yaw rate
)
≫
0
corr(curl,yaw rate)≫0

corr
(
magnitude
,
speed
)
≫
0
corr(magnitude,speed)≫0

Scaling plan

One-camera smoke test (most front-facing cam). Implement end-to-end with overlays + validation scatter.

Multi-camera aggregation with cosine weights.

Ablations: ROI radius ∈ {40,60,80}, FLOW_ALGO ∈ {Farnebäck, TV-L1}.

Optional: multi-plane warps (ground + walls) or offline mesh render if needed.

2) Full script outline (single file)
"""
vis_features_from_virtual_head_view.py

Derives visual covariates (optic-flow geometry + motion-energy PCs) that approximate
retinal consequences of self-motion using scene cameras, 3D head pose, and a calibrated
floor plane. Designed for static arena + static cameras (only the monkey moves).

I/O
---
Inputs:
  - bins.csv                 # columns: bin_idx, t_prev, t_curr (sec), ...
  - head_pose.parquet        # per-bin: P_head_world[x,y,z], R_head_world[3x3] or f_head_world[x,y,z]
  - kinematics.parquet       # per-bin speeds: forward_speed, yaw_rate, speed (for validation)
  - calib/{cam}.json         # per camera: K (3x3), R (3x3), t (3,), optional distCoeffs
  - floor_plane.json         # plane in world coords: a,b,c,d (prefer upright.plane)
  - frames/{cam}/            # timestamped frames or a video + index

Outputs:
  - vis_features.parquet     # per bin: flow_mag, flow_div, flow_curl, flow_dir_x, flow_dir_y,
                             #          me_pc1..me_pcK, wsum, cams_used
  - diagnostics/             # optional overlays, validation plots

Conventions:
  - Camera model: X_c = R X_w + t ; pixel p ~ K X_c
  - Camera forward in world: f_cam = R^T [0,0,1]
  - Plane Π: n^T X + d = 0  (n and d must share scale; if n normalized, rescale d accordingly)
"""

from dataclasses import dataclass
import json, math, pathlib, numpy as np, pandas as pd
from typing import Dict, Tuple, List, Optional

# =========================
# Config & CLI
# =========================

@dataclass
class Config:
    cams: List[str]
    patch_hw: Tuple[int,int] = (200, 200)          # (H,W) in pixels
    mpp: float = 0.005                              # meters-per-pixel on the plane (5 mm/px)
    roi_radius_px: int = 60
    flow_algo: str = "farneback"                    # or "tvl1"
    L_ahead_m: float = 1.0
    zscore_per_session: bool = True
    n_me_pcs: int = 6
    # File paths
    bins_csv: str = "bins.csv"
    head_pose_parquet: str = "head_pose.parquet"
    kinematics_parquet: str = "kinematics.parquet"
    floor_plane_json: str = "floor_plane.json"
    calib_dir: str = "calib"
    frames_root: str = "frames"
    out_parquet: str = "vis_features.parquet"
    diagnostics_dir: str = "diagnostics"

# =========================
# Geometry utils
# =========================

def load_plane(json_path: str) -> Tuple[np.ndarray, float]:
    J = json.loads(pathlib.Path(json_path).read_text())
    P = J.get("upright", {}).get("plane", J["plane"])
    n = np.array([P["a"], P["b"], P["c"]], dtype=float)
    d = float(P["d"])
    return n, d

def cam_forward_world(R: np.ndarray) -> np.ndarray:
    return R.T @ np.array([0.0, 0.0, 1.0])

def visibility_weight(f_head: np.ndarray, R: np.ndarray) -> float:
    return float(max(0.0, np.dot(f_head, cam_forward_world(R))))

def ray_plane_intersect(P: np.ndarray, v: np.ndarray, n: np.ndarray, d: float) -> np.ndarray:
    denom = float(n @ v)
    if abs(denom) < 1e-8:
        return np.full(3, np.nan)
    s = -(n @ P + d) / denom
    return P + s * v

def head_tangent_basis_on_plane(f_head: np.ndarray, n: np.ndarray) -> Tuple[np.ndarray,np.ndarray]:
    # Project head-forward onto plane, normalize, then lateral = n x e1
    f_proj = f_head - (np.dot(n, f_head) / (np.dot(n, n))) * n
    if np.linalg.norm(f_proj) < 1e-8:
        # rare: head-forward ≈ plane normal; fall back to any orthonormal pair in plane
        f_proj = np.array([1.0,0.0,0.0])
    e1 = f_proj / np.linalg.norm(f_proj)
    e2 = np.cross(n, e1); e2 /= np.linalg.norm(e2)
    return e1, e2

def build_patch_world_grid(Qw, e1, e2, H, W, mpp):
    cy, cx = (H-1)/2.0, (W-1)/2.0
    J, I = np.meshgrid(np.arange(W), np.arange(H))   # cols (x), rows (y)
    Xw = (Qw[:,None,None]
          + ((J - cx)*mpp)[None,:,:]*e1[:,None,None]
          + ((I - cy)*mpp)[None,:,:]*e2[:,None,None])     # (3,H,W)
    return Xw

def project_world_to_image(Xw, K, R, t):
    Xc = R @ Xw.reshape(3,-1) + t[:,None]
    z = Xc[2]; z[z<=1e-8] = np.nan
    u = K[0,0]*Xc[0]/z + K[0,2]
    v = K[1,1]*Xc[1]/z + K[1,2]
    return u.reshape(Xw.shape[1:]), v.reshape(Xw.shape[1:])

# =========================
# Image/flow utils (stubs)
# =========================

def read_frame_nearest(frames_index, t_sec) -> Optional[np.ndarray]:
    """Return grayscale float32 image at nearest timestamp; None if missing."""
    # Implement via per-camera time index → frame path/video seek
    ...

def bilinear_sample(img: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Sample img at (u,v) with bilinear interpolation; out-of-bounds → NaN."""
    ...

def dense_flow(prev_img: np.ndarray, curr_img: np.ndarray, algo="farneback") -> Tuple[np.ndarray,np.ndarray]:
    """Return (u,v) flow fields for two same-sized grayscale images."""
    ...

def disk_mask(H, W, cy, cx, r) -> np.ndarray:
    Y, X = np.ogrid[:H, :W]
    return (Y-cy)**2 + (X-cx)**2 <= r*r

def flow_stats(u, v, mask) -> Tuple[float,float,float,float,float]:
    mag = np.hypot(u, v)
    mag_mean = np.nanmean(mag[mask])
    # Smooth slightly before gradients if noisy (optional)
    du_dx, du_dy = np.gradient(u)
    dv_dx, dv_dy = np.gradient(v)
    div  = np.nanmean((du_dx + dv_dy)[mask])
    curl = np.nanmean((dv_dx - du_dy)[mask])
    vx = np.nanmean(u[mask]); vy = np.nanmean(v[mask])
    norm = np.hypot(vx, vy) + 1e-8
    return float(mag_mean), float(div), float(curl), float(vx/norm), float(vy/norm)

# =========================
# Motion-energy (ME) bank
# =========================

def build_gabor_bank() -> List[np.ndarray]:
    """Return small set of 2D Gabors for orientations × scales × wavelengths."""
    ...

def me_energy_vector(img_prev, img_curr, mask, gabors) -> np.ndarray:
    diff = (img_curr - img_prev)
    feats = []
    for g in gabors:
        resp = conv2d(diff, g)          # implement or use scipy.signal.fftconvolve
        e = np.nanmean((resp**2)[mask])
        feats.append(e)
    return np.asarray(feats, dtype=np.float32)

# =========================
# Main per-camera feature extraction
# =========================

def features_for_camera(cam_id: str, cfg: Config, plane, frames_index, K,R,t,
                        bins_df, head_df) -> pd.DataFrame:
    n, d = plane
    H,W = cfg.patch_hw
    gabors = build_gabor_bank()
    rows = []
    me_rows = []
    me_bin_idxs = []

    for row in bins_df.itertuples():
        bin_idx, t_prev, t_curr = row.bin_idx, row.t_prev, row.t_curr
        P = head_df.loc[bin_idx, ["x","y","z"]].to_numpy(float)
        f = head_df.loc[bin_idx, ["fx","fy","fz"]].to_numpy(float)
        f /= np.linalg.norm(f) + 1e-8

        w_vis = visibility_weight(f, R)
        if w_vis <= 0:     # camera behind the head
            rows.append((bin_idx, np.nan, np.nan, np.nan, np.nan, np.nan, w_vis))
            continue

        Qw = ray_plane_intersect(P, f, n, d)
        if not np.isfinite(Qw).all():
            rows.append((bin_idx, np.nan, np.nan, np.nan, np.nan, np.nan, w_vis))
            continue

        e1, e2 = head_tangent_basis_on_plane(f, n)
        Xw = build_patch_world_grid(Qw, e1, e2, H, W, cfg.mpp)
        u, v = project_world_to_image(Xw, K, R, t)

        prev = read_frame_nearest(frames_index, t_prev)
        curr = read_frame_nearest(frames_index, t_curr)
        if prev is None or curr is None:
            rows.append((bin_idx, np.nan, np.nan, np.nan, np.nan, np.nan, w_vis))
            continue

        patch_prev = bilinear_sample(prev, u, v)
        patch_curr = bilinear_sample(curr, u, v)
        if np.isnan(patch_prev).mean() > 0.2 or np.isnan(patch_curr).mean() > 0.2:
            rows.append((bin_idx, np.nan, np.nan, np.nan, np.nan, np.nan, w_vis))
            continue

        # Flow stats
        fu, fv = dense_flow(patch_prev, patch_curr, algo=cfg.flow_algo)
        cy, cx = (H-1)/2.0, (W-1)/2.0
        mask = disk_mask(H, W, cy, cx, cfg.roi_radius_px)
        mag, div, curl, dx, dy = flow_stats(fu, fv, mask)
        rows.append((bin_idx, mag, div, curl, dx, dy, w_vis))

        # ME vector (store for PCA post-hoc)
        me_vec = me_energy_vector(patch_prev, patch_curr, mask, gabors)
        me_rows.append(me_vec); me_bin_idxs.append(bin_idx)

    feats = pd.DataFrame(rows, columns=[
        "bin_idx","flow_mag","flow_div","flow_curl","flow_dir_x","flow_dir_y","w_vis"
    ]).set_index("bin_idx")

    if len(me_rows):
        ME = pd.DataFrame(np.vstack(me_rows), index=me_bin_idxs)
    else:
        ME = pd.DataFrame()

    return feats, ME

# =========================
# Multi-camera aggregation
# =========================

def aggregate_across_cameras(feats_per_cam: Dict[str,pd.DataFrame]) -> pd.DataFrame:
    # Weighted average by w_vis, track wsum and cams_used
    all_idx = sorted(set().union(*[df.index for df in feats_per_cam.values()]))
    cols = ["flow_mag","flow_div","flow_curl","flow_dir_x","flow_dir_y"]
    out = pd.DataFrame(index=all_idx, columns=cols + ["wsum","cams_used"], dtype=float)
    out[:] = np.nan

    for idx in all_idx:
        num = np.zeros(len(cols), dtype=float)
        wsum = 0.0; used = 0
        for df in feats_per_cam.values():
            if idx not in df.index: continue
            w = df.at[idx,"w_vis"]
            if not np.isfinite(w) or w <= 0: continue
            v = df.loc[idx, cols].to_numpy(float)
            if not np.isfinite(v).all(): continue
            num += w * v; wsum += w; used += 1
        if wsum > 0:
            out.loc[idx, cols] = num / wsum
            out.loc[idx, "wsum"] = wsum
            out.loc[idx, "cams_used"] = used
    return out

# =========================
# Motion-energy PCA
# =========================

def pca_fit_transform(ME_concat: pd.DataFrame, n_pcs: int) -> Tuple[pd.DataFrame, np.ndarray]:
    # z-score columns, SVD
    X = ME_concat.to_numpy(np.float32)
    mu = np.nanmean(X, axis=0, keepdims=True); sd = np.nanstd(X, axis=0, keepdims=True)+1e-8
    Xz = (X - mu) / sd
    # drop rows with NaN
    good = np.isfinite(Xz).all(axis=1)
    U,S,Vt = np.linalg.svd(Xz[good], full_matrices=False)
    VtK = Vt[:n_pcs]
    scores = np.full((Xz.shape[0], n_pcs), np.nan, dtype=np.float32)
    scores[good] = Xz[good] @ VtK.T
    pc_df = pd.DataFrame(scores, index=ME_concat.index, columns=[f"me_pc{i+1}" for i in range(n_pcs)])
    return pc_df, VtK  # keep VtK if you want to project future sessions identically

# =========================
# Validation plots (optional stubs)
# =========================

def validate(agg_df: pd.DataFrame, kin_df: pd.DataFrame, out_dir: str):
    """Compute correlations: div↔forward_speed, curl↔yaw_rate, mag↔speed. Save plots."""
    ...

# =========================
# Main
# =========================

def main(cfg: Config):
    n, d = load_plane(cfg.floor_plane_json)

    bins_df = pd.read_csv(cfg.bins_csv)  # bin_idx, t_prev, t_curr (sec)
    bins_df = bins_df.sort_values("bin_idx")

    head_df = pd.read_parquet(cfg.head_pose_parquet)  # columns: x,y,z, fx,fy,fz (unit)
    # Ensure alignment: head_df indexed by bin_idx
    head_df = head_df.set_index("bin_idx").loc[bins_df.bin_idx]

    # Optional: kinematics for validation
    kin = pd.read_parquet(cfg.kinematics_parquet).set_index("bin_idx").loc[bins_df.bin_idx]

    feats_per_cam = {}
    ME_per_cam = {}

    for cam in cfg.cams:
        # Load calibration
        C = json.loads(pathlib.Path(cfg.calib_dir, f"{cam}.json").read_text())
        K = np.asarray(C["K"], dtype=float)
        R = np.asarray(C["R"], dtype=float)
        t = np.asarray(C["t"], dtype=float)
        # Build per-camera frame index (timestamp → file/seek); user-defined
        frames_index = build_frames_index(pathlib.Path(cfg.frames_root, cam))
        feats, ME = features_for_camera(cam, cfg, (n,d), frames_index, K,R,t, bins_df, head_df)
        feats_per_cam[cam] = feats
        if ME is not None and not ME.empty:
            ME_per_cam[cam] = ME

    agg = aggregate_across_cameras(feats_per_cam)

    # Motion-energy: concatenate by index, compute PCs, then merge back (NaN where missing)
    if len(ME_per_cam):
        # You can aggregate ME across cams first (weighted) OR compute PCs on concatenated energies.
        # Here: concatenate energies across cams (stack by columns).
        ME_concat = None
        for cam, ME in ME_per_cam.items():
            ME = ME.add_prefix(f"{cam}_")
            ME_concat = ME if ME_concat is None else ME_concat.join(ME, how="outer")
        me_pc_df, VtK = pca_fit_transform(ME_concat, cfg.n_me_pcs)
        vis = agg.join(me_pc_df, how="left")
    else:
        vis = agg

    # Z-score per session (recommended for encoding)
    if cfg.zscore_per_session:
        for col in vis.columns:
            m = np.nanmean(vis[col]); s = np.nanstd(vis[col]) + 1e-8
            vis[col] = (vis[col] - m) / s

    vis.reset_index().to_parquet(cfg.out_parquet, index=False)

    # Validation
    try:
        validate(vis, kin, cfg.diagnostics_dir)
    except Exception as e:
        print("Validation skipped:", e)

if __name__ == "__main__":
    # Example single-camera smoke test:
    cfg = Config(cams=["Cam_001"], roi_radius_px=60, flow_algo="farneback")
    main(cfg)

Notes & caveats (practical)

Lens distortion: rectify frames or incorporate distortion in projection; otherwise small warps near edges can corrupt flow.

Depth & occlusion: ground-plane warp ignores vertical parallax; that’s fine for a conservative VIS regressor. Add wall planes later if needed.

Time mapping: no look-ahead—map bin 
[
𝑡
−
1
,
𝑡
]
[t−1,t] to nearest frames; drop bins with missing frames.

Gradients: lightly smooth flow before 
∇
∇ to stabilize divergence/curl; keep the same for all ablations.

Weights: clip negatives in 
𝑤
=
max
⁡
(
0
,
cos
⁡
(
∠
(
𝑓
head
,
𝑓
cam
)
)
)
w=max(0,cos(∠(f
head
	​

,f
cam
	​

))); record wsum, cams_used.

Ablations: run 2×3 grid (algo × ROI) and show VIS-only / KIN-only / VIS+KIN explained variance heatmap.

Encoding/variance analysis checklist (to be convincing)

Model: Ridge/Poisson GLM (depending on your spike rate transform), 5-fold CV, time-lags on VIS and KIN (e.g., −200…+200 ms).

Partition: Report 
𝑅
2
_
VIS
R
2
_VIS, 
𝑅
2
_
KIN
R
2
_KIN, 
𝑅
2
_
VIS+KIN
R
2
_VIS+KIN and unique contributions (partial 
𝑅
2
R
2
) via nested CV.

Controls: time-shifted VIS (circular shuffle), static-patch VIS (no head warp) to show those fail, confirming necessity of head-centric warp.

Sanity: show div↔forward speed, curl↔yaw rate correlations within the same bins used for encoding.