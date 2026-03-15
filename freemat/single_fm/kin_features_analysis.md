Let’s lock down a lean, defensible KIN feature set that matches your Figure 2 plan and plays nicely with VIS.

1) KIN feature set (core, per bin)

Goal: capture self-motion geometry in a chosen reference frame (head / torso / world). Keep it compact to avoid soaking up VIS variance via overparameterization.

Linear (from COM position 
𝑃
(
𝑡
)
P(t))

𝑉
(
𝑡
)
=
𝑑
𝑃
𝑑
𝑡
V(t)=
dt
dP
	​

 (central diff, light smooth)

Express in frame 
𝐹
∈
{
head
,
torso
,
world
}
F∈{head,torso,world}: 
𝑣
𝐹
=
𝑅
𝐹
←
𝑊
⊤
𝑉
v
F
	​

=R
F←W
⊤
	​

V

v_|| = forward (x-axis of frame)

v_\perp = lateral (y-axis)

v_z = vertical (z-axis; plane normal)

Optionals (if cheap, but not required for Fig 2): 
∣
∣
𝑉
∣
∣
∣∣V∣∣ (speed), a_|| etc. I’d start without acceleration to keep KIN tight.

Angular (from orientation 
𝑅
𝐹
(
𝑡
)
R
F
	​

(t))

Yaw angle 
𝜓
𝐹
(
𝑡
)
ψ
F
	​

(t) from 
𝑅
𝐹
(
𝑡
)
R
F
	​

(t) around up (plane normal), then:

yaw_rate 
=
𝑑
𝜓
𝐹
𝑑
𝑡
=
dt
dψ
F
	​

	​

 (captures turns)

delta_yaw_{head–torso} 
=
𝜓
head
−
𝜓
torso
=ψ
head
	​

−ψ
torso
	​

 (head–body dissociation; static, not a rate)

Allocentric position (POS) for the FULL model variants

(X,Y) on floor plane (2D thin-plate spline) or (X,Y,Z) if vertical matters. Use POS only in KIN/POS and FULL, not in VIS-only.

So, minimal KIN-core features (per frame & lag):

[v_parallel, v_perp, v_z, yaw_rate, delta_yaw_head_torso]

2) Reference frames (how to build 
𝑅
𝐹
←
𝑊
R
F←W
	​

)

Use your floor upright.

World frame (W): identity.

Head frame (H):

Forward: 
𝑒
𝑥
=
proj
Π
(
𝑓
^
head
)
e
x
	​

=proj
Π
	​

(
f
^
	​

head
	​

), normalize.

Up: 
𝑒
𝑧
=
𝑛
^
Π
e
z
	​

=
n
^
Π
	​

 (plane normal, unit).

Left: 
𝑒
𝑦
=
𝑒
𝑧
×
𝑒
𝑥
e
y
	​

=e
z
	​

×e
x
	​

.
Set 
𝑅
𝑊
←
𝐻
=
[
𝑒
𝑥
,
𝑒
𝑦
,
𝑒
𝑧
]
R
W←H
	​

=[e
x
	​

,e
y
	​

,e
z
	​

] as columns; then 
𝑅
𝐻
←
𝑊
=
𝑅
𝑊
←
𝐻
⊤
R
H←W
	​

=R
W←H
⊤
	​

.

Torso frame (T): replace 
𝑓
^
head
f
^
	​

head
	​

 with a torso-forward proxy (e.g., mid-hip→neck projected to plane; or shoulder-line×up).

3) Lags

Build KIN (and POS) at 
𝜏
∈
{
−
800
,
…
,
+
800
}
τ∈{−800,…,+800} ms (bin shifts). Keep VIS at 
𝜏
=
0
τ=0.

For each 
(
frame
,
𝜏
)
(frame,τ), evaluate KIN-only and FULL.

4) Preprocess

Smoothing: COM positions → 1–3 bin moving average before derivs; same for yaw (unwrap → diff → smooth).

Derivatives: central differences, respect edges.

Z-score within session for all predictors (fit scalers on train folds only).

Drop bins where VIS is NaN; apply the same mask to KIN/POS to keep rows aligned.

5) Controls

Time-shift VIS by ±10–60 s → VIS-only should crumble; FULL shouldn’t spuriously rise.

Orthogonalization (train-only): regress KIN on VIS in train folds → use KIN residuals to verify KIN-unique stability (report as a robustness check).

Ablations: FLOW_ALGO, ROI radius; expect minimal movement in conclusions.

6) GAM formulas (per neuron)

Use cubic splines (k≈8–12) for 1D, thin-plate for 2D.

VIS-only

FR
∼
𝑠
(
flow_mag
)
+
𝑠
(
flow_div
)
+
𝑠
(
flow_curl
)
+
𝑠
(
flow_dir_x
)
+
𝑠
(
flow_dir_y
)
+
∑
𝑘
𝑠
(
me_pc
𝑘
)
FR∼s(flow_mag)+s(flow_div)+s(flow_curl)+s(flow_dir_x)+s(flow_dir_y)+∑
k
	​

s(me_pck)

KIN-only (frame, lag)

FR
∼
𝑠
(
𝑣
∥
)
+
𝑠
(
𝑣
⊥
)
+
𝑠
(
𝑣
𝑧
)
+
𝑠
(
yaw_rate
)
+
𝑠
(
Δ
yaw
𝐻
−
𝑇
)
FR∼s(v
∥
	​

)+s(v
⊥
	​

)+s(v
z
	​

)+s(yaw_rate)+s(Δyaw
H−T
	​

)
(+ optionally 
𝑠
(
𝑋
,
𝑌
)
s(X,Y) or 
𝑠
(
𝑋
,
𝑌
,
𝑍
)
s(X,Y,Z) for POS variants)

FULL = VIS + KIN (+ POS)

Unique variances:

KIN-unique
=
𝑅
2
(
FULL
)
−
𝑅
2
(
VIS
)
KIN-unique=R
2
(FULL)−R
2
(VIS);

VIS-unique
=
𝑅
2
(
FULL
)
−
𝑅
2
(
KIN
)
VIS-unique=R
2
(FULL)−R
2
(KIN).
Compute on the same held-out bins (blocked 5-fold CV per session).

7) Pseudocode (feature builder)
def kin_features(pose, plane, frame: str):
    # pose: with P(t) [Nx3], head_dir(t) [Nx3], torso_dir(t) [Nx3] (or derive)
    n = unit(plane.normal)
    P = pose.com                      # (N,3)
    dt = 1.0 / pose.fps
    V = central_diff(P, dt)           # (N,3)

    def axes_from(dir_vec):
        ex = unit(project_to_plane(dir_vec, n))   # forward
        ez = n                                    # up
        ey = unit(np.cross(ez, ex))               # left
        R_W_to_F = np.stack([ex, ey, ez], axis=1) # cols
        return R_W_to_F.T                          # F<-W

    if frame == 'head':
        R = axes_from(pose.head_dir)              # (N,3,3)
        psi_head = unwrap_yaw(R, n)               # (N,)
        psi_torso = unwrap_yaw(axes_from(pose.torso_dir), n)
    elif frame == 'torso':
        R = axes_from(pose.torso_dir)
        psi_head = unwrap_yaw(axes_from(pose.head_dir), n)
        psi_torso = unwrap_yaw(R, n)
    else:  # world
        R = np.broadcast_to(np.eye(3), (len(P),3,3))
        psi_head = unwrap_yaw(axes_from(pose.head_dir), n)
        psi_torso = unwrap_yaw(axes_from(pose.torso_dir), n)

    vF = bmm_vec(R, V)                 # (N,3) -> [v_parallel, v_perp, v_z]
    yaw_rate = central_diff(psi_head, dt)
    delta_yaw_ht = wrap_pi(psi_head - psi_torso)

    return {
        "v_parallel": vF[:,0],
        "v_perp":     vF[:,1],
        "v_z":        vF[:,2],
        "yaw_rate":   yaw_rate,
        "delta_yaw_head_torso": delta_yaw_ht,
        # POS (world XY/XYZ) if requested:
        "X": P[:,0], "Y": P[:,1], "Z": P[:,2],
    }

8) Design matrix & CV (per neuron)
# Build master table
T = align_bins(spikes, VIS, KIN, mask=~VIS.isna().any(axis=1))
T = zscore_within_session(T, by="session_id", cols=predictor_cols)

# Blocked CV indices (5 contiguous blocks per session)
folds = make_blocked_folds(T["session_id"], T["t_bin_s"], n_folds=5)

for frame in ["head","torso","world"]:
    KIN_F = make_all_lags(kin_features_frame[frame], lags_ms=range(-800, 801, bin_ms))
    for tau in lags_ms:
        X_vis = VIS.values
        X_kin = KIN_F[tau].values

        # Fit GAMs on train folds; transform (splines) fit on train only
        r2_vis  = cv_r2(GAM_VIS.fit(X_vis[train], y[train]),
                        X_vis[test], y[test])
        r2_kin  = cv_r2(GAM_KIN.fit(X_kin[train], y[train]),
                        X_kin[test], y[test])
        r2_full = cv_r2(GAM_FULL.fit(np.c_[X_vis,X_kin][train], y[train]),
                        np.c_[X_vis,X_kin][test], y[test])

        kin_unique = r2_full - r2_vis
        vis_unique = r2_full - r2_kin
        log_result(neuron_id, frame, tau, r2_vis, r2_kin, r2_full, kin_unique, vis_unique)

9) Pitfalls to avoid (a few you’ll thank yourself for)

Leakage: fit spline bases, z-scores, PCA (for ME), and any orthogonalization on train only, apply to test.

Lag shifting: when you shift KIN by +τ, drop rows that would peek into the future.

Yaw unwrap before differencing; otherwise spikes of ±2π will haunt your curl.

Balanced folds: ensure each fold has bins from all sessions; don’t split mid-burst if spikes are ultra-sparse—use long contiguous blocks.

10) Reporting (what to put in the figure captions)

Median [95% CI] cv-
𝑅
2
R
2
 for VIS / KIN / FULL.

Median [IQR] KIN-unique and VIS-unique; per-bank distributions.

Best frame histogram (expect head > torso ≫ world).

Best lag distribution (expect +100–300 ms).

Controls: VIS time-shift crashes VIS-only; FULL stable; orthogonalized KIN still carries unique variance.