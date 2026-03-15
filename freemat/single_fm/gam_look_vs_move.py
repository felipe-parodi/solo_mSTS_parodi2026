"""

### gam encoding script that:

- loads aligned neural and pose Data
- computes features on the fly
- for now we're going to compute: looking vs forward movement

this script will test three questions:
1. do mid-STS neurons code facing direction independent of movement?
    - AKAK looking-only while still: mask speed < s0 (based off of distribution?)
2. movement-only controlling for look (unique variance of movement vs facing angle)
3. alignment effect: does firing depend on moving into the gaze vs. away?


plots to make:
- tuning to theta_look at low speed
- partial for v_parallel and align_cos
- scatter phi vs FR

We'll use a GAM with 5-fold blocked CV

we want to report:
- disentangling panel
    - stillness: tuning to theta_look for exemplar units; % units with significant look tuning and by dorsal/ventral bank
    - movement: partials for v_parallel, align_cos
- proxy VIS control
    - VIS-proxy-only vs KIN-lite vs FULL-proxy CV-R2 and partial R2
    - KIN-unique = FULL - VIS - proxy 
"""

Script layout (one file: gam_stepA_look_vs_move.py)
0) Imports & config

numpy, pandas, matplotlib

pyGAM (preferred) or statsmodels fallback

small helpers for blocked CV, z-scoring, Fourier basis

1) Load data

Inputs (aligned, per-bin):

neural.parquet: columns → session_id, t_bin_s, unit_id, fr (or spikes→zFR you prefer), bank (dorsal/ventral)

pose.parquet: t_bin_s, P_x,P_y,P_z, fhead_x,fhead_y,fhead_z, n_x,n_y,n_z
(floor normal n_* can be constant per session)

Join on session_id, t_bin_s. Drop rows with missing.

2) Compute features (on the fly)

For each row:

In-plane basis: pick stable plane axes (e1,e2) ⟂ n_hat once per session.

Project:

f_proj = fhead - (fhead·n_hat) n_hat; normalize to f_hat

V = dP/dt (central diff per session); also V_proj = V - (V·n_hat) n_hat

Angles:

theta_look = atan2(f_hat·e2, f_hat·e1) (unwrap per session)

theta_move = atan2(V_proj·e2, V_proj·e1) (unwrap per session)

KIN-lite:

speed = ||V||

head axes: ex = f_hat, ey = n_hat × ex, ez = n_hat

v_parallel = V·ex, v_perp = V·ey, v_z = V·ez

yaw_rate = d(theta_look)/dt (central diff on unwrapped)

align_cos = cos(theta_move − theta_look)

Fourier for periodic look tuning (order K=3 works well):
fourier(theta_look) = [sin θ, cos θ, sin 2θ, cos 2θ, sin 3θ, cos 3θ]

Per-session z-score predictors on train folds only (apply to test) — speed, v_* , yaw_rate, align_cos, and each Fourier column.

Threshold for “still”

Pick a robust cut:

s0 = np.quantile(speed_in_session, 0.15)   # ~15th percentile
still_mask = speed < s0


(Report s0 per session in QC.)

3) Blocked 5-fold CV

Per session, split t_bin_s into 5 contiguous blocks; interleave across sessions to form folds with similar sizes. Return fold_id per row.

4) Models (per unit)

Use pyGAM.LinearGAM if available; otherwise fallback to statsmodels OLS with cubic B-splines for continuous terms and raw Fourier columns for angle.

M1 — Looking-only while still

Data: filter still_mask.
Design: X = [fourier(theta_look), yaw_rate]
Model: GAM with smooth on yaw_rate only (Fourier are linear columns).

Metric: CV-R² (held-out).

Optional null: block-wise circular time shift of FR within session (e.g., 100 perms) → p-value.

M2 — Movement-only controlling for look

All bins.
X = [fourier(theta_look), s(v_parallel), s(v_perp), s(v_z), s(yaw_rate)]

Report: CV-R²; partial R² for movement block versus look (nested CV):

Fit LOOK: FR ~ Fourier(θ_look) + s(yaw_rate)

Fit MOVE+LOOK: add s(v_parallel), s(v_perp), s(v_z)

move_unique = R²(MOVE+LOOK) − R²(LOOK)

M3 — Alignment effect

All bins.
X = [s(speed), s(align_cos), s(yaw_rate)]
(If using pyGAM, you can also add interaction te(speed, align_cos) later.)

Report partials for v_parallel (from M2) and align_cos (from M3).

For tomorrow: skip heavy permutations; mark “significant” as CV-R² > 0.01 and/or move_unique > 0.005. You can add shuffles later.

5) Plots

Tuning to θ_look at low speed (per exemplar unit):

Sweep θ from −π…π, set yaw_rate=0; predict with M1; plot periodic curve.

Partials:

gam.partial_dependence(term_of_v_parallel) in M2,

gam.partial_dependence(term_of_align_cos) in M3.

Scatter: yaw_rate (φ) vs FR (thin hexbin + loess or GAM smooth).

6) Reporting

Aggregate by unit and by bank:

% units with M1 CV-R² > 0.01 (“look-tuned while still”), stratified by dorsal/ventral.

Medians [95% CI bootstrap] of:

M1 CV-R² (look)

M2 CV-R² (move+look)

move_unique from M2

M3 partial effect size (Δŷ range) for align_cos

(You can stub the “proxy VIS control” section for later—commented out hooks in code.)

Minimal code skeleton (runnable with pyGAM; elide I/O details)
# gam_stepA_look_vs_move.py
import numpy as np, pandas as pd
from dataclasses import dataclass
from pygam import LinearGAM, s
import matplotlib.pyplot as plt

# ---------- helpers ----------
def blocked_folds(df, n_folds=5):
    df = df.sort_values(['session_id','t_bin_s']).copy()
    df['fold'] = -1
    for sid, g in df.groupby('session_id'):
        n = len(g); idx = g.index.to_numpy()
        cuts = np.array_split(idx, n_folds)
        for k, ids in enumerate(cuts):
            df.loc[ids, 'fold'] = k
    return df['fold'].astype(int)

def fourier_theta(theta, K=3):
    cols = []
    for k in range(1, K+1):
        cols += [np.sin(k*theta), np.cos(k*theta)]
    return np.column_stack(cols)

def zscore_by_session(train_idx, all_idx, X, sess):
    Xz = X.copy()
    for sid in np.unique(sess):
        m = (sess==sid)
        t = m & train_idx
        mu = np.nanmean(Xz[t], axis=0)
        sd = np.nanstd(Xz[t], axis=0); sd[sd<1e-8]=1.0
        Xz[m] = (Xz[m]-mu)/sd
    return Xz

# ---------- features ----------
def compute_features(df):
    # assumes columns: P_x,P_y,P_z, fhead_x..z, n_x..z, session_id, t_bin_s
    P = df[['P_x','P_y','P_z']].to_numpy(float)
    f = df[['fhead_x','fhead_y','fhead_z']].to_numpy(float)
    n = df[['n_x','n_y','n_z']].to_numpy(float)
    n = (n.T/np.linalg.norm(n,axis=1)).T
    # stable plane basis
    t = np.array([1,0,0],float)
    e1 = np.cross(n, t); bad = (np.linalg.norm(e1,axis=1)<1e-6)
    e1[bad] = np.cross(n[bad], np.array([0,1,0],float))
    e1 = (e1.T/np.linalg.norm(e1,axis=1)).T
    e2 = np.cross(n, e1)

    # project head forward to plane
    f_proj = f - (np.sum(f*n,axis=1,keepdims=True))*n
    fhat = (f_proj.T/np.linalg.norm(f_proj,axis=1)).T
    # head axes
    ex = fhat
    ez = n
    ey = np.cross(ez, ex); ey = (ey.T/np.linalg.norm(ey,axis=1)).T

    # derivatives per session (central diff)
    V = np.zeros_like(P)
    for sid, g in df.groupby('session_id'):
        idx = g.index.to_numpy()
        dt = np.gradient(df.loc[idx,'t_bin_s'].to_numpy())
        G = P[idx]
        V[idx][1:-1] = (G[2:]-G[:-2])/(dt[1:-1][:,None]+1e-12)
        V[idx][0] = (G[1]-G[0])/(dt[0]+1e-12)
        V[idx][-1] = (G[-1]-G[-2])/(dt[-1]+1e-12)

    speed = np.linalg.norm(V, axis=1)
    vpar = np.sum(V*ex, axis=1)
    vperp= np.sum(V*ey, axis=1)
    vz   = np.sum(V*ez, axis=1)

    # angles
    def atan2v(a,b): return np.arctan2(np.sum(a*e2,1), np.sum(a*e1,1))
    theta_look = np.unwrap(atan2v(fhat, fhat))  # uses fhat components
    Vproj = V - (np.sum(V*n,axis=1,keepdims=True))*n
    theta_move = np.unwrap(atan2v(Vproj, Vproj))
    yaw_rate = np.gradient(theta_look, df['t_bin_s'])

    align_cos = np.cos(theta_move - theta_look)

    out = df[['session_id','unit_id','bank','t_bin_s']].copy()
    out['speed']=speed; out['v_parallel']=vpar; out['v_perp']=vperp; out['v_z']=vz
    out['yaw_rate']=yaw_rate; out['theta_look']=theta_look; out['align_cos']=align_cos
    return out

# ---------- model runners ----------
def cv_r2(y, yhat, groups):
    # compute R2 per fold then average
    sst = np.var(y, ddof=0)
    return 1.0 - np.mean((y - yhat)**2)/ (sst+1e-12)

def fit_predict_gam(X, y, folds, smooth_cols):
    yhat = np.zeros_like(y, dtype=float)
    for k in range(5):
        train = folds!=k; test = ~train
        # leakage-safe scaling
        Xz = zscore_by_session(train, np.ones_like(train, bool), X, sess)
        gam = LinearGAM(sum(s(i) for i in smooth_cols)).fit(Xz[train], y[train])
        yhat[test] = gam.predict(Xz[test])
    return yhat

# ---------- main ----------
def run(models=('M1','M2','M3')):
    neural = pd.read_parquet('neural.parquet')
    pose   = pd.read_parquet('pose.parquet')
    df = neural.merge(pose, on=['session_id','t_bin_s'], how='inner')
    feats = compute_features(df)
    T = df[['session_id','unit_id','bank','t_bin_s','fr']].join(feats.set_index(['session_id','unit_id','t_bin_s']), on=['session_id','unit_id','t_bin_s'])
    # folds
    T['fold'] = blocked_folds(T[['session_id','t_bin_s']].drop_duplicates().merge(T[['session_id','t_bin_s']], how='right'))
    results = []
    for uid, G in T.groupby('unit_id'):
        sess = G['session_id'].to_numpy()
        folds = G['fold'].to_numpy()
        y = G['fr'].to_numpy(float)

        # STILL mask
        s0 = np.quantile(G['speed'], 0.15)
        m_still = G['speed'] < s0

        # Fourier(theta_look)
        F = fourier_theta(G['theta_look'].to_numpy(), K=3)

        # M1: looking-only while still
        if 'M1' in models and m_still.sum()>100:
            X1 = np.column_stack([F[m_still], G.loc[m_still,'yaw_rate']])
            y1 = y[m_still]
            sess1 = sess[m_still]; folds1 = folds[m_still]
            # scale per session; smooth yaw_rate only (last column)
            smooth_cols = [X1.shape[1]-1]
            # pack sess globally for scaler
            globals()['sess'] = sess1
            yhat1 = fit_predict_gam(X1, y1, folds1, smooth_cols)
            r2_look = cv_r2(y1, yhat1, folds1)
        else:
            r2_look = np.nan

        # M2: movement-only controlling for look
        X2 = np.column_stack([F, G[['yaw_rate','v_parallel','v_perp','v_z']].to_numpy(float)])
        smooth_cols = list(range(F.shape[1], X2.shape[1]))  # smooth kinetic terms
        globals()['sess'] = sess
        yhat2 = fit_predict_gam(X2, y, folds, smooth_cols)
        r2_move_look = cv_r2(y, yhat2, folds)

        # LOOK baseline for partial R2
        Xlook = np.column_stack([F, G[['yaw_rate']].to_numpy(float)])
        smooth_cols_look = [F.shape[1]]  # yaw_rate only
        globals()['sess'] = sess
        yhatL = fit_predict_gam(Xlook, y, folds, smooth_cols_look)
        r2_look_all = cv_r2(y, yhatL, folds)
        move_unique = r2_move_look - r2_look_all

        # M3: alignment effect
        X3 = G[['speed','align_cos','yaw_rate']].to_numpy(float)
        smooth_cols3 = [0,1,2]
        globals()['sess'] = sess
        yhat3 = fit_predict_gam(X3, y, folds, smooth_cols3)
        r2_align = cv_r2(y, yhat3, folds)

        results.append({'unit_id':uid,'bank':G['bank'].iloc[0],
                        'r2_look_still':r2_look,
                        'r2_move_plus_look':r2_move_look,
                        'r2_look_all':r2_look_all,
                        'move_unique':move_unique,
                        'r2_align':r2_align})
    R = pd.DataFrame(results)
    R.to_csv('stepA_results.csv', index=False)
    # quick report
    ok = R['r2_look_still'] > 0.01
    by_bank = R.groupby('bank')['r2_look_still'].apply(lambda s: (s>0.01).mean())
    print('Look tuning (still) % units:', float((ok).mean()))
    print('By bank:\n', by_bank)

if __name__ == '__main__':
    run()

Plot stubs (add after run()):

plot_theta_tuning(unit_id) → build θ grid, predict with M1, line plot.

plot_partial(unit_id, term='v_parallel') → use M2’s GAM partial_dependence.

plot_scatter_yaw(unit_id) → hexbin of yaw_rate vs FR with smooth overlay.

TL;DR

Features: v_parallel, v_perp, v_z, yaw_rate, theta_look (→ Fourier), align_cos, speed.

Models: M1 (look while still), M2 (movement unique vs look), M3 (alignment).

CV: 5 blocked folds; per-session scaling fit on train, apply to test.

Outputs: CV-R² per model; move_unique; % look-tuned units overall & by bank; exemplar plots.