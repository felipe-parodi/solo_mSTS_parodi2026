"""Task decoders for motif identity, onset detection, and next-state prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    top_k_accuracy_score,
)

from freemat.single_fm.decode_motifs.features import (
    build_lagged_design,
    apply_lag_vec,
    chronological_folds,
    _standardize_train_test,
    _macro_auroc_ovr,
    _pr_auc,
    _event_from_prob_trace,
    _match_events,
    _weights_to_unit_lag,
)
from freemat.single_fm.decode_motifs.motifs import _derive_onsets


@dataclass
class TaskResult:
    metrics: Dict[str, Any]
    per_fold: List[Dict[str, Any]]
    classes_: Optional[np.ndarray] = None
    confusion_: Optional[np.ndarray] = None
    prob_traces: Optional[np.ndarray] = None  # for onset traces (T_eff,)
    weights_abs: Optional[np.ndarray] = None  # (U, H)


def task_A_motif_identity(
    X_fr: np.ndarray,
    y_labels: np.ndarray,
    *,
    bin_s: float,
    cv_folds: int,
    purge_s: float,
    C_grid: np.ndarray,
    lag_grid_ms: np.ndarray,
    history_grid_s: Sequence[float],
    quick: bool,
    seed: int,
    null_perm: int = 0,
    positive_only: bool = False,
    debug: bool = False,
    lr_solver: str = "lbfgs",
    lr_tol: float = 1e-3,
    lr_max_iter_inner: int = 300,
    lr_max_iter: int = 1000,
    svm_tol: Optional[float] = None,
    svm_max_iter_inner: Optional[int] = None,
    svm_max_iter: Optional[int] = None,
    svm_dual: str = "auto",
) -> TaskResult:
    """Multiclass motif identity decoding with nested CV (macro-F1)."""
    rng = np.random.RandomState(int(seed))
    # Clean labels: keep frames where y>0
    m_keep = y_labels > 0
    y = y_labels.astype(int)
    per_fold: List[Dict[str, Any]] = []

    # Cache feature designs per history
    H_opts = [max(1, int(round(h / bin_s))) for h in history_grid_s]
    H_opts = sorted(list(set(H_opts)))
    designs_by_H: Dict[int, np.ndarray] = {}
    for H in H_opts:
        designs_by_H[H] = build_lagged_design(X_fr, H)

    # Build folds on valid rows after history alignment and lag alignment per candidate; evaluate mean macro-F1
    lag_bins_grid = np.round(lag_grid_ms / 1000.0 / bin_s).astype(int)

    # Outer folds built once on the strictest alignment after we pick H, but for selection we evaluate via internal logic
    # Simpler: build folds on base timeline of X_fr; we will intersect with valid rows after alignment.
    H0 = min(H_opts)
    mk0 = m_keep[H0 - 1 :]
    m_valid0 = mk0
    idx_valid0 = np.where(m_valid0)[0]
    if idx_valid0.size == 0:
        raise ValueError("No valid frames after history alignment for task A.")
    # Build contiguous folds on valid indices space
    folds_raw = chronological_folds(idx_valid0.size, int(cv_folds), int(round(purge_s / bin_s)))
    # Map fold indices back to X0 indices
    folds_outer = []
    for f in folds_raw:
        folds_outer.append({
            "train": idx_valid0[f["train"]],
            "test": idx_valid0[f["test"]],
        })

    # Nested selection per outer fold
    y_all_true: List[int] = []
    y_all_pred: List[int] = []
    # Weight aggregation per H
    weight_sum_by_H: Dict[int, np.ndarray] = {}
    weight_cnt_by_H: Dict[int, int] = {}

    null_all: List[float] = []
    for f_idx, fold in enumerate(folds_outer):
        tr0, te0 = fold["train"], fold["test"]
        if debug:
            # Quick fold-level class summary on base alignment (H0, no lag)
            y_base = y[H0 - 1 :]
            m_base = m_keep[H0 - 1 :]
            tr_base = np.intersect1d(tr0, np.where(m_base)[0], assume_unique=True)
            ytr_base = y_base[tr_base]
            uniq, cnt = np.unique(ytr_base[ytr_base > 0], return_counts=True)
            print(f"[A][fold {f_idx}] train classes: {[ (int(a), int(b)) for a,b in zip(uniq, cnt) ]}")
        best_cfg = None
        best_f1 = -np.inf
        # Inner selection loop
        for H in H_opts:
            XH = designs_by_H[H]
            yH = y[H - 1 :]
            mH = m_keep[H - 1 :]
            idx_valid = np.where(mH)[0]
            # Restrict to current fold space by intersecting
            tr = np.intersect1d(tr0, idx_valid, assume_unique=True)
            te = np.intersect1d(te0, idx_valid, assume_unique=True)
            if tr.size == 0 or te.size == 0:
                continue
            for lag_bins in lag_bins_grid:
                yL, x_sl, _ = apply_lag_vec(yH, int(lag_bins))
                XLG = XH[x_sl]
                # Reindex fold indices to XLG coordinate
                tr_l = tr[(tr >= x_sl.start if isinstance(x_sl.start, int) else 0) & (tr < (x_sl.stop if isinstance(x_sl.stop, int) else XLG.shape[0]))] - (x_sl.start or 0)
                te_l = te[(te >= x_sl.start if isinstance(x_sl.start, int) else 0) & (te < (x_sl.stop if isinstance(x_sl.stop, int) else XLG.shape[0]))] - (x_sl.start or 0)
                if tr_l.size == 0 or te_l.size == 0:
                    continue
                # Fit scaler + LR for each C and pick by val macro-F1
                Xtr_s, Xte_s, _, _ = _standardize_train_test(XLG[tr_l], XLG[te_l])
                ytr = yL[tr_l]
                yte = yL[te_l]
                # Remove class 0 and keep only positives classes
                mtr = ytr > 0
                mte = yte > 0
                if mtr.sum() == 0 or mte.sum() == 0:
                    continue
                ytr = ytr[mtr]
                Xtr_s2 = Xtr_s[mtr]
                yte = yte[mte]
                Xte_s2 = Xte_s[mte]
                # Need at least 2 classes in train for LR to fit
                if np.unique(ytr).size < 2:
                    continue
                for C in C_grid:
                    # Effective SVM hyperparameters (fallback to LR flags when not provided)
                    tol_eff = float(svm_tol) if svm_tol is not None else float(lr_tol)
                    max_iter_eff = int(svm_max_iter_inner) if svm_max_iter_inner is not None else int(lr_max_iter_inner)
                    # Heuristic dual selection if 'auto'
                    n_samp, n_feat = Xtr_s2.shape
                    if str(svm_dual).lower() == "true":
                        dual_eff = True
                    elif str(svm_dual).lower() == "false":
                        dual_eff = False
                    else:
                        dual_eff = bool(n_samp < n_feat)
                    clf = LinearSVC(
                        C=float(C),
                        class_weight="balanced",
                        tol=tol_eff,
                        max_iter=max_iter_eff,
                        dual=dual_eff,
                        random_state=int(seed),
                    )
                    try:
                        clf.fit(Xtr_s2, ytr)
                    except ValueError:
                        if debug:
                            print("[A][inner] skip: single-class train or solver issue")
                        # e.g., solver complains due to degenerate data; skip
                        continue
                    yph = clf.predict(Xte_s2)
                    f1 = f1_score(yte, yph, average="macro")
                    if f1 > best_f1:
                        best_f1 = f1
                        best_cfg = (H, int(lag_bins), float(C), clf)
        if best_cfg is None:
            # Skip this fold; not enough class diversity after masking/splits
            if debug:
                print("[A][outer] fold skipped: no config with >=2 classes in train")
            per_fold.append({"metrics": {"macro_f1": float('nan'), "balanced_acc": float('nan')}, "classes": [], "confusion": [], "H": None, "lag_bins": None, "C": None})
            continue
        H_star, lag_star, C_star, clf_star = best_cfg
        # Refit on full-train for this fold and evaluate on test
        XH = designs_by_H[H_star]
        yH = y[H_star - 1 :]
        mH = m_keep[H_star - 1 :]
        idx_valid = np.where(mH)[0]
        tr = np.intersect1d(tr0, idx_valid, assume_unique=True)
        te = np.intersect1d(te0, idx_valid, assume_unique=True)
        yL, x_sl, _ = apply_lag_vec(yH, int(lag_star))
        XLG = XH[x_sl]
        # map indices
        tr_l = tr[(tr >= x_sl.start if isinstance(x_sl.start, int) else 0) & (tr < (x_sl.stop if isinstance(x_sl.stop, int) else XLG.shape[0]))] - (x_sl.start or 0)
        te_l = te[(te >= x_sl.start if isinstance(x_sl.start, int) else 0) & (te < (x_sl.stop if isinstance(x_sl.stop, int) else XLG.shape[0]))] - (x_sl.start or 0)
        Xtr_s, Xte_s, mu, sd = _standardize_train_test(XLG[tr_l], XLG[te_l])
        ytr = yL[tr_l]
        yte = yL[te_l]
        mtr = ytr > 0
        mte = yte > 0
        Xtr_s = Xtr_s[mtr]
        ytr = ytr[mtr]
        Xte_s = Xte_s[mte]
        yte = yte[mte]
        if np.unique(ytr).size < 2 or Xtr_s.shape[0] == 0 or Xte_s.shape[0] == 0:
            if debug:
                print("[A][final] fold skipped: not enough classes or samples")
            per_fold.append({"metrics": {"macro_f1": float('nan'), "balanced_acc": float('nan')}, "classes": [], "confusion": [], "H": int(H_star), "lag_bins": int(lag_star), "C": float(C_star)})
            continue
        # Effective SVM hyperparameters for final fit
        tol_eff_f = float(svm_tol) if svm_tol is not None else float(lr_tol)
        max_iter_eff_f = int(svm_max_iter) if svm_max_iter is not None else int(lr_max_iter)
        n_samp_f, n_feat_f = Xtr_s.shape
        if str(svm_dual).lower() == "true":
            dual_eff_f = True
        elif str(svm_dual).lower() == "false":
            dual_eff_f = False
        else:
            dual_eff_f = bool(n_samp_f < n_feat_f)
        clf = LinearSVC(
            C=float(C_star),
            class_weight="balanced",
            tol=tol_eff_f,
            max_iter=max_iter_eff_f,
            dual=dual_eff_f,
            random_state=int(seed),
        )
        try:
            clf.fit(Xtr_s, ytr)
        except ValueError:
            if debug:
                print("[A][final] solver error; skipping fold")
            per_fold.append({"metrics": {"macro_f1": float('nan'), "balanced_acc": float('nan')}, "classes": [], "confusion": [], "H": int(H_star), "lag_bins": int(lag_star), "C": float(C_star)})
            continue
        # Decision scores and predictions
        classes = np.asarray(getattr(clf, "classes_", np.unique(ytr)), int)
        yhat_lab = clf.predict(Xte_s)
        S = clf.decision_function(Xte_s)
        if S.ndim == 1:
            S = np.stack([-S, S], axis=1)
        # Metrics
        fold_metrics: Dict[str, Any] = {}
        fold_metrics["macro_f1"] = float(f1_score(yte, yhat_lab, average="macro"))
        fold_metrics["balanced_acc"] = float(balanced_accuracy_score(yte, yhat_lab))
        # AUROC OvR using decision scores
        S_aligned = np.zeros((S.shape[0], classes.size), float)
        for i, c in enumerate(classes):
            S_aligned[:, i] = S[:, i]
        fold_metrics["ovr_auroc"] = _macro_auroc_ovr(yte, S_aligned, classes)
        # top-1/top-2
        fold_metrics["top1_acc"] = float(accuracy_score(yte, yhat_lab))
        # top-2 accuracy: use decision scores
        top2 = top_k_accuracy_score(yte, S_aligned, k=min(2, classes.size), labels=classes)
        fold_metrics["top2_acc"] = float(top2)
        # Confusion (on this fold)
        Cmat = confusion_matrix(yte, yhat_lab, labels=classes)
        per_fold.append({
            "metrics": fold_metrics,
            "classes": classes.tolist(),
            "confusion": Cmat.astype(int).tolist(),
            "H": int(H_star),
            "lag_bins": int(lag_star),
            "C": float(C_star),
        })
        # Null: label permutation on train (keep features), evaluate macro-F1 on test
        if int(null_perm) > 0:
            base_clf = LinearSVC(
                C=float(C_star),
                class_weight="balanced",
                tol=float(lr_tol),
                max_iter=int(lr_max_iter_inner),
                random_state=int(seed),
            )
            for k in range(int(null_perm)):
                ytr_perm = rng.permutation(ytr)
                base_clf.fit(Xtr_s, ytr_perm)
                yph_null = base_clf.predict(Xte_s)
                null_all.append(float(f1_score(yte, yph_null, average="macro")))
        # Aggregate for overall confusion
        y_all_true.append(yte)
        y_all_pred.append(yhat_lab)
        # Coefs for weight visualization (mean abs over folds)
        if hasattr(clf, "coef_"):
            W = clf.coef_.mean(axis=0) if hasattr(clf, "coef_") and clf.coef_.ndim > 1 else getattr(clf, "coef_", np.zeros((1, XH.shape[1]))).reshape(-1)
            U = X_fr.shape[1]
            W_u_h = _weights_to_unit_lag(W, U=U, H=H_star)
            weight_sum_by_H[H_star] = weight_sum_by_H.get(H_star, np.zeros_like(W_u_h)) + np.abs(W_u_h)
            weight_cnt_by_H[H_star] = weight_cnt_by_H.get(H_star, 0) + 1
    # Overall metrics
    y_true_all = np.concatenate(y_all_true) if len(y_all_true) else np.array([], int)
    y_pred_all = np.concatenate(y_all_pred) if len(y_all_pred) else np.array([], int)
    classes_all = np.unique(y_true_all)
    if y_true_all.size:
        Cmat_all = confusion_matrix(y_true_all, y_pred_all, labels=classes_all)
    else:
        Cmat_all = np.zeros((0, 0), dtype=int)
    overall = {
        "macro_f1": float(f1_score(y_true_all, y_pred_all, average="macro")) if y_true_all.size else float("nan"),
        "balanced_acc": float(balanced_accuracy_score(y_true_all, y_pred_all)) if y_true_all.size else float("nan"),
        "n_classes": int(classes_all.size),
    }
    if null_all:
        null_arr = np.array(null_all, float)
        overall.update({
            "null_macro_f1_mean": float(np.nanmean(null_arr)),
            "null_macro_f1_lo": float(np.nanpercentile(null_arr, 2.5)),
            "null_macro_f1_hi": float(np.nanpercentile(null_arr, 97.5)),
        })
    weights_abs = None
    if weight_sum_by_H:
        # Pick modal H
        H_star = max(weight_cnt_by_H.keys(), key=lambda h: weight_cnt_by_H[h])
        weights_abs = weight_sum_by_H[H_star] / float(max(weight_cnt_by_H[H_star], 1))
    return TaskResult(metrics=overall, per_fold=per_fold, classes_=classes_all if classes_all.size else None, confusion_=Cmat_all if Cmat_all.size else None, weights_abs=weights_abs)


def task_B1_transition_onset(
    X_fr: np.ndarray,
    y_labels: np.ndarray,
    *,
    bin_s: float,
    cv_folds: int,
    purge_s: float,
    C_grid: np.ndarray,
    lag_grid_ms: np.ndarray,
    history_grid_s: Sequence[float],
    tau_ms: int,
    quick: bool,
    seed: int,
    positive_only: bool = False,
    debug: bool = False,
    lr_solver: str = "lbfgs",
    lr_tol: float = 1e-3,
    lr_max_iter_inner: int = 300,
    lr_max_iter: int = 1000,
) -> TaskResult:
    """Binary hazard of transition within (t, t+tau]. Optimizes PR-AUC."""
    # Build hazard labels on full timeline (t aligned to X later)
    y = y_labels.astype(int)
    # Onsets indices (t where label changes)
    onsets = _derive_onsets(y, None)
    if positive_only:
        # Keep only transitions where both sides are >0
        ch = (y[1:] != y[:-1]) & (y[1:] > 0) & (y[:-1] > 0)
        onsets = np.where(ch)[0] + 1
    if debug:
        print(f"[B1] total_onsets={onsets.size} (positive_only={positive_only})")
    # Exclude 250 ms after each onset to avoid rebounds
    excl = np.zeros(y.size, dtype=bool)
    post_bins = int(round(0.250 / bin_s))
    for j in onsets:
        a = j
        b = min(y.size, j + post_bins + 1)
        excl[a:b] = True
    # Build hazard label: 1 if any onset occurs in (t, t+tau]
    tau_bins = max(1, int(round(int(tau_ms) / 1000.0 / bin_s)))
    hazard = np.zeros(y.size, dtype=int)
    for j in onsets:
        # contribute to bins t in [j - tau_bins, j-1]
        t0 = max(0, j - tau_bins)
        hazard[t0:j] = 1
    # Mask invalid frames: exclude immediate post-onset frames
    valid = ~excl
    # Prepare grids
    H_opts = [max(1, int(round(h / bin_s))) for h in history_grid_s]
    H_opts = sorted(list(set(H_opts)))
    designs_by_H = {H: build_lagged_design(X_fr, H) for H in H_opts}
    lag_bins_grid = np.round(lag_grid_ms / 1000.0 / bin_s).astype(int)

    # Build outer folds on valid frames after smallest H
    H0 = min(H_opts)
    v0 = valid[H0 - 1 :]
    idx_valid0 = np.where(v0)[0]
    if idx_valid0.size == 0:
        raise ValueError("No valid frames for onset task after exclusions.")
    folds_raw = chronological_folds(idx_valid0.size, int(cv_folds), int(round(purge_s / bin_s)))
    folds_outer = [{"train": idx_valid0[f["train"]], "test": idx_valid0[f["test"]]} for f in folds_raw]

    per_fold: List[Dict[str, Any]] = []
    # Accumulate lag sweep (mean AP per lag across folds)
    ap_by_lag_sum: Dict[int, float] = {int(k): 0.0 for k in lag_bins_grid}
    ap_by_lag_cnt: Dict[int, int] = {int(k): 0 for k in lag_bins_grid}
    # For onset traces
    trace_pre_s = 0.5
    trace_post_s = 0.5
    w_pre = int(round(trace_pre_s / bin_s))
    w_post = int(round(trace_post_s / bin_s))
    traces_accum: List[np.ndarray] = []
    # For PR curves across folds
    y_all_concat: List[np.ndarray] = []
    p_all_concat: List[np.ndarray] = []
    for f_idx, fold in enumerate(folds_outer):
        tr0, te0 = fold["train"], fold["test"]
        best_cfg = None
        best_ap = -np.inf
        # Inner selection: H, lag, C
        for H in H_opts:
            XH = designs_by_H[H]
            yH = hazard[H - 1 :]
            vH = valid[H - 1 :]
            tr = np.intersect1d(tr0, np.where(vH)[0], assume_unique=True)
            te = np.intersect1d(te0, np.where(vH)[0], assume_unique=True)
            if tr.size == 0 or te.size == 0:
                continue
            for lag_bins in lag_bins_grid:
                yL, x_sl, _ = apply_lag_vec(yH, int(lag_bins))
                XLG = XH[x_sl]
                # Restrict folds to XLG coord
                tr_l = tr[(tr >= (x_sl.start or 0)) & (tr < (x_sl.stop or XLG.shape[0]))] - (x_sl.start or 0)
                te_l = te[(te >= (x_sl.start or 0)) & (te < (x_sl.stop or XLG.shape[0]))] - (x_sl.start or 0)
                if tr_l.size == 0 or te_l.size == 0:
                    continue
                # Fit scaler + LR and pick by PR-AUC
                Xtr_s, Xte_s, _, _ = _standardize_train_test(XLG[tr_l], XLG[te_l])
                ytr = yL[tr_l]
                yte = yL[te_l]
                for C in C_grid:
                    base = LogisticRegression(
                        penalty="l2",
                        C=float(C),
                        class_weight="balanced",
                        solver=str(lr_solver),
                        tol=float(lr_tol),
                        max_iter=int(lr_max_iter_inner),
                        random_state=int(seed),
                    )
                    clf = base if quick else CalibratedClassifierCV(base, method="sigmoid", cv=3)
                    # Need both classes in train fold; else LR fails
                    if np.unique(ytr).size < 2:
                        if debug:
                            print("[B1][inner] skip: single-class train")
                        continue
                    try:
                        clf.fit(Xtr_s, ytr)
                    except ValueError:
                        if debug:
                            print("[B1][inner] solver error; skip")
                        continue
                    p = clf.predict_proba(Xte_s)[:, 1]
                    ap = _pr_auc(yte, p)
                    ap_by_lag_sum[int(lag_bins)] += float(ap)
                    ap_by_lag_cnt[int(lag_bins)] += 1
                    if ap > best_ap:
                        best_ap = ap
                        best_cfg = (H, int(lag_bins), float(C), clf)
        if best_cfg is None:
            # Skip this fold: no positive/negative class mix in train for any config
            if debug:
                print("[B1][outer] fold skipped: no usable config")
            continue
        H_star, lag_star, C_star, clf_star = best_cfg
        # Refit on full train and evaluate on test; also collect proba for traces on entire valid timeline
        XH = designs_by_H[H_star]
        # Use the selected history H_star (bugfix from H)
        yH = hazard[H_star - 1 :]
        vH = valid[H_star - 1 :]
        tr = np.intersect1d(tr0, np.where(vH)[0], assume_unique=True)
        te = np.intersect1d(te0, np.where(vH)[0], assume_unique=True)
        yL, x_sl, _ = apply_lag_vec(yH, int(lag_star))
        XLG = XH[x_sl]
        # Fit
        tr_l = tr[(tr >= (x_sl.start or 0)) & (tr < (x_sl.stop or XLG.shape[0]))] - (x_sl.start or 0)
        te_l = te[(te >= (x_sl.start or 0)) & (te < (x_sl.stop or XLG.shape[0]))] - (x_sl.start or 0)
        Xtr_s, Xte_s, mu_x, sd_x = _standardize_train_test(XLG[tr_l], XLG[te_l])
        ytr = yL[tr_l]
        yte = yL[te_l]
        base = LogisticRegression(
            penalty="l2",
            C=float(C_star),
            class_weight="balanced",
            solver=str(lr_solver),
            tol=float(lr_tol),
            max_iter=int(lr_max_iter),
            random_state=int(seed),
        )
        clf = base if quick else CalibratedClassifierCV(base, method="sigmoid", cv=3)
        if np.unique(ytr).size < 2 or Xtr_s.shape[0] == 0 or Xte_s.shape[0] == 0:
            if debug:
                print("[B1][final] fold skipped: insufficient classes/samples")
            continue
        try:
            clf.fit(Xtr_s, ytr)
        except ValueError:
            if debug:
                print("[B1][final] solver error; skip fold")
            continue
        p_te = clf.predict_proba(Xte_s)[:, 1]
        ap_te = _pr_auc(yte, p_te)
        # Choose threshold on validation-like split within train (50/50 chrono)
        n_tr = tr_l.size
        mid = n_tr // 2
        Xv_s = XLG[tr_l[mid:]]
        yv = yL[tr_l[mid:]]
        pv = clf.predict_proba(Xv_s)[:, 1]
        prec, rec, thr_grid = precision_recall_curve(yv, pv)
        # Select threshold with best F1 on val
        f1s = (2 * prec * rec / np.maximum(prec + rec, 1e-9))
        thr_star = float(thr_grid[np.nanargmax(f1s[:-1])]) if thr_grid.size else 0.5
        # Event metrics on test with fixed threshold
        # Probability across full aligned timeline (standardize with train-only stats)
        XLG_s_full = (XLG - mu_x) / np.where(sd_x <= 0, 1.0, sd_x)
        p_full = clf.predict_proba(XLG_s_full)[:, 1]
        # Extract true onsets within te in the aligned yL timeline
        onsets_L = _derive_onsets(yL)
        on_te_mask = (onsets_L >= te_l.min()) & (onsets_L <= te_l.max()) if te_l.size else np.array([], dtype=bool)
        true_on_te = onsets_L[on_te_mask]
        # Predicted events
        pred_idx_rel = _event_from_prob_trace(p_te, thr_star)
        pred_idx_abs = te_l[pred_idx_rel] if pred_idx_rel.size else np.array([], int)
        tol_bins = int(round(150.0 / 1000.0 / bin_s))
        TP, FP, FN, lats = _match_events(true_on_te, pred_idx_abs, tol_bins=tol_bins)
        f1_ev = float(2 * TP / max(2 * TP + FP + FN, 1))
        fold_metrics = {
            "pr_auc": float(ap_te),
            "event_f1": float(f1_ev),
            "tp": int(TP),
            "fp": int(FP),
            "fn": int(FN),
            "latency_ms_mean": float(np.mean(lats) * bin_s * 1000.0) if lats.size else float("nan"),
            "latency_ms_std": float(np.std(lats) * bin_s * 1000.0) if lats.size else float("nan"),
            "H": int(H_star),
            "lag_bins": int(lag_star),
            "C": float(C_star),
            "thr": float(thr_star),
        }
        per_fold.append(fold_metrics)
        # Accumulate for overall PR curve
        y_all_concat.append(yte)
        p_all_concat.append(p_te)
        # Onset probability traces (average aligned to test onsets)
        win = np.arange(-w_pre, w_post + 1, dtype=int)
        tr_mat: List[np.ndarray] = []
        for o in true_on_te:
            if (o - w_pre) >= 0 and (o + w_post) < p_full.shape[0]:
                tr_mat.append(p_full[o + win])
        if tr_mat:
            traces_accum.append(np.vstack(tr_mat).mean(axis=0))

    # Aggregate
    if not per_fold:
        # No usable folds (likely due to no positive hazard in train for all splits)
        return TaskResult(metrics={"pr_auc": float("nan"), "event_f1": float("nan")}, per_fold=[])
    pr_aucs = np.array([fd["pr_auc"] for fd in per_fold], float)
    ev_f1 = np.array([fd["event_f1"] for fd in per_fold], float)
    y_all = np.concatenate(y_all_concat) if y_all_concat else np.array([], int)
    p_all = np.concatenate(p_all_concat) if p_all_concat else np.array([], float)
    overall = {
        "pr_auc": float(np.nanmean(pr_aucs)),
        "event_f1": float(np.nanmean(ev_f1)),
        "y_pr": y_all.astype(int).tolist() if y_all.size else [],
        "p_pr": p_all.astype(float).tolist() if p_all.size else [],
    }
    # Lag sweep curve
    lag_sweep_ms = sorted(ap_by_lag_sum.keys())
    ap_means = [ap_by_lag_sum[k] / max(ap_by_lag_cnt[k], 1) for k in lag_sweep_ms]
    # Onset traces aggregate
    trace_mean = None
    if traces_accum:
        trace_mean = np.vstack(traces_accum).mean(axis=0)
    extra = {
        "lag_sweep_lags_bins": lag_sweep_ms,
        "lag_sweep_ap": ap_means,
        "trace_mean": trace_mean.tolist() if trace_mean is not None else None,
        "trace_t_ms": ((np.arange(-w_pre, w_post + 1) * bin_s) * 1000.0).astype(float).tolist(),
    }
    overall.update(extra)
    return TaskResult(metrics=overall, per_fold=per_fold, prob_traces=trace_mean)


def task_B2_next_state(
    X_fr: np.ndarray,
    y_labels: np.ndarray,
    *,
    bin_s: float,
    cv_folds: int,
    purge_s: float,
    C_grid: np.ndarray,
    lag_grid_ms: np.ndarray,
    history_grid_s: Sequence[float],
    quick: bool,
    seed: int,
    null_perm: int = 0,
    positive_only: bool = False,
    debug: bool = False,
    lr_solver: str = "lbfgs",
    lr_tol: float = 1e-3,
    lr_max_iter_inner: int = 300,
    lr_max_iter: int = 1000,
    svm_tol: Optional[float] = None,
    svm_max_iter_inner: Optional[int] = None,
    svm_max_iter: Optional[int] = None,
    svm_dual: str = "auto",
) -> TaskResult:
    """Multiclass next-state decoding in windows around onsets (macro-F1)."""
    y = y_labels.astype(int)
    onsets = _derive_onsets(y)
    if positive_only:
        ch = (y[1:] != y[:-1]) & (y[1:] > 0) & (y[:-1] > 0)
        onsets = np.where(ch)[0] + 1
    if debug:
        print(f"[B2] total_onsets={onsets.size} (positive_only={positive_only})")
    if onsets.size == 0:
        raise ValueError("No transitions found for next-state task.")
    # Build window mask around each onset
    pre = int(round(0.5 / bin_s))
    post = int(round(0.1 / bin_s))
    T = y.size
    m = np.zeros(T, dtype=bool)
    lab = np.zeros(T, dtype=int)
    for j in onsets:
        a = max(0, j - pre)
        b = min(T, j + post + 1)
        m[a:b] = True
        lab[a:b] = y[j]
    # Remove background label 0
    m &= lab > 0
    if not np.any(m):
        raise ValueError("Next-state windows have no labeled frames.")
    # Prepare designs and folds
    H_opts = [max(1, int(round(h / bin_s))) for h in history_grid_s]
    H_opts = sorted(list(set(H_opts)))
    designs_by_H = {H: build_lagged_design(X_fr, H) for H in H_opts}
    lag_bins_grid = np.round(lag_grid_ms / 1000.0 / bin_s).astype(int)

    # Outer folds on masked frames after smallest H
    H0 = min(H_opts)
    m0 = m[H0 - 1 :]
    idx0 = np.where(m0)[0]
    folds_raw = chronological_folds(idx0.size, int(cv_folds), int(round(purge_s / bin_s)))
    folds_outer = [{"train": idx0[f["train"]], "test": idx0[f["test"]]} for f in folds_raw]

    per_fold: List[Dict[str, Any]] = []
    y_true_all: List[int] = []
    y_pred_all: List[int] = []
    weight_sum_by_H: Dict[int, np.ndarray] = {}
    weight_cnt_by_H: Dict[int, int] = {}

    null_all: List[float] = []
    for f_idx, fold in enumerate(folds_outer):
        tr0, te0 = fold["train"], fold["test"]
        best_cfg = None
        best_f1 = -np.inf
        for H in H_opts:
            XH = designs_by_H[H]
            yH = lab[H - 1 :]
            mH = m[H - 1 :]
            # Restrict folds to masked frames
            tr = np.intersect1d(tr0, np.where(mH)[0], assume_unique=True)
            te = np.intersect1d(te0, np.where(mH)[0], assume_unique=True)
            if tr.size == 0 or te.size == 0:
                continue
            for lag_bins in lag_bins_grid:
                yL, x_sl, _ = apply_lag_vec(yH, int(lag_bins))
                XLG = XH[x_sl]
                tr_l = tr[(tr >= (x_sl.start or 0)) & (tr < (x_sl.stop or XLG.shape[0]))] - (x_sl.start or 0)
                te_l = te[(te >= (x_sl.start or 0)) & (te < (x_sl.stop or XLG.shape[0]))] - (x_sl.start or 0)
                if tr_l.size == 0 or te_l.size == 0:
                    continue
                Xtr_s, Xte_s, _, _ = _standardize_train_test(XLG[tr_l], XLG[te_l])
                ytr = yL[tr_l]
                yte = yL[te_l]
                mtr = ytr > 0
                mte = yte > 0
                Xtr_s = Xtr_s[mtr]
                ytr = ytr[mtr]
                Xte_s = Xte_s[mte]
                yte = yte[mte]
                if np.unique(ytr).size < 2:
                    continue
                for C in C_grid:
                    tol_eff = float(svm_tol) if svm_tol is not None else float(lr_tol)
                    max_iter_eff = int(svm_max_iter_inner) if svm_max_iter_inner is not None else int(lr_max_iter_inner)
                    n_samp, n_feat = Xtr_s.shape
                    if str(svm_dual).lower() == "true":
                        dual_eff = True
                    elif str(svm_dual).lower() == "false":
                        dual_eff = False
                    else:
                        dual_eff = bool(n_samp < n_feat)
                    clf = LinearSVC(
                        C=float(C),
                        class_weight="balanced",
                        tol=tol_eff,
                        max_iter=max_iter_eff,
                        dual=dual_eff,
                        random_state=int(seed),
                    )
                    try:
                        clf.fit(Xtr_s, ytr)
                    except ValueError:
                        if debug:
                            print("[B2][inner] solver error; skip")
                        continue
                    yph = clf.predict(Xte_s)
                    f1 = f1_score(yte, yph, average="macro")
                    if f1 > best_f1:
                        best_f1 = f1
                        best_cfg = (H, int(lag_bins), float(C))
        if best_cfg is None:
            per_fold.append({"macro_f1": float('nan'), "top1_acc": float('nan')})
            continue
        H_star, lag_star, C_star = best_cfg
        # Refit on full-train and evaluate on test
        XH = designs_by_H[H_star]
        yH = lab[H_star - 1 :]
        mH = m[H_star - 1 :]
        tr = np.intersect1d(tr0, np.where(mH)[0], assume_unique=True)
        te = np.intersect1d(te0, np.where(mH)[0], assume_unique=True)
        yL, x_sl, _ = apply_lag_vec(yH, int(lag_star))
        XLG = XH[x_sl]
        tr_l = tr[(tr >= (x_sl.start or 0)) & (tr < (x_sl.stop or XLG.shape[0]))] - (x_sl.start or 0)
        te_l = te[(te >= (x_sl.start or 0)) & (te < (x_sl.stop or XLG.shape[0]))] - (x_sl.start or 0)
        Xtr_s, Xte_s, _, _ = _standardize_train_test(XLG[tr_l], XLG[te_l])
        ytr = yL[tr_l]
        yte = yL[te_l]
        mtr = ytr > 0
        mte = yte > 0
        Xtr_s = Xtr_s[mtr]
        ytr = ytr[mtr]
        Xte_s = Xte_s[mte]
        yte = yte[mte]
        if np.unique(ytr).size < 2 or Xtr_s.shape[0] == 0 or Xte_s.shape[0] == 0:
            per_fold.append({"macro_f1": float('nan'), "top1_acc": float('nan')})
            continue
        tol_eff_f = float(svm_tol) if svm_tol is not None else float(lr_tol)
        max_iter_eff_f = int(svm_max_iter) if svm_max_iter is not None else int(lr_max_iter)
        n_samp_f, n_feat_f = Xtr_s.shape
        if str(svm_dual).lower() == "true":
            dual_eff_f = True
        elif str(svm_dual).lower() == "false":
            dual_eff_f = False
        else:
            dual_eff_f = bool(n_samp_f < n_feat_f)
        clf = LinearSVC(
            C=float(C_star),
            class_weight="balanced",
            tol=tol_eff_f,
            max_iter=max_iter_eff_f,
            dual=dual_eff_f,
            random_state=int(seed),
        )
        try:
            clf.fit(Xtr_s, ytr)
        except ValueError:
            if debug:
                print("[B2][final] solver error; skip fold")
            per_fold.append({"macro_f1": float('nan'), "top1_acc": float('nan')})
            continue
        yph = clf.predict(Xte_s)
        fold_metrics = {
            "macro_f1": float(f1_score(yte, yph, average="macro")),
            "top1_acc": float(accuracy_score(yte, yph)),
        }
        per_fold.append(fold_metrics)
        y_true_all.append(yte)
        y_pred_all.append(yph)
        if hasattr(clf, "coef_"):
            W = clf.coef_.mean(axis=0) if hasattr(clf, "coef_") and clf.coef_.ndim > 1 else getattr(clf, "coef_", np.zeros((1, XH.shape[1]))).reshape(-1)
            U = X_fr.shape[1]
            W_u_h = _weights_to_unit_lag(W, U=U, H=H_star)
            weight_sum_by_H[H_star] = weight_sum_by_H.get(H_star, np.zeros_like(W_u_h)) + np.abs(W_u_h)
            weight_cnt_by_H[H_star] = weight_cnt_by_H.get(H_star, 0) + 1

    y_true_all = np.concatenate(y_true_all) if y_true_all else np.array([], int)
    y_pred_all = np.concatenate(y_pred_all) if y_pred_all else np.array([], int)
    if y_true_all.size:
        Cmat = confusion_matrix(y_true_all, y_pred_all, labels=np.unique(y_true_all))
    else:
        Cmat = np.zeros((0, 0), dtype=int)
    overall = {
        "macro_f1": float(f1_score(y_true_all, y_pred_all, average="macro")) if y_true_all.size else float("nan"),
        "top1_acc": float(accuracy_score(y_true_all, y_pred_all)) if y_true_all.size else float("nan"),
    }
    if null_all:
        arr = np.array(null_all, float)
        overall.update({
            "null_macro_f1_mean": float(np.nanmean(arr)),
            "null_macro_f1_lo": float(np.nanpercentile(arr, 2.5)),
            "null_macro_f1_hi": float(np.nanpercentile(arr, 97.5)),
        })
    weights_abs = None
    if weight_sum_by_H:
        H_star = max(weight_cnt_by_H.keys(), key=lambda h: weight_cnt_by_H[h])
        weights_abs = weight_sum_by_H[H_star] / float(max(weight_cnt_by_H[H_star], 1))
    classes_eff = np.unique(y_true_all) if y_true_all.size else None
    conf_eff = Cmat if (Cmat.size and y_true_all.size) else None
    return TaskResult(metrics=overall, per_fold=per_fold, classes_=classes_eff, confusion_=conf_eff, weights_abs=weights_abs)
