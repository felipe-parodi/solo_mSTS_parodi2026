from __future__ import annotations

"""Visual feature primitives: optic flow and motion energy in an ROI.

These are generic helpers used to compute per-frame or per-pair image features
inside a circular region-of-interest, typically a head-anchored projection in
scene cameras.
"""

from typing import Dict, Optional, Sequence, Tuple
import math
import numpy as np


def bilinear_sample(img: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Bilinear sample a single-channel image at floating-point pixel coords.

    Args:
        img: Grayscale image array of shape (H, W) or (H, W, 1).
        u: Array of x-coordinates (columns), same shape as `v`.
        v: Array of y-coordinates (rows), same shape as `u`.

    Returns:
        Array of sampled values with the same shape as `u`/`v`. Out-of-bounds
        coordinates are set to NaN.

    Notes:
        - For RGB images, pass a pre-converted grayscale image to avoid
          ambiguous channel handling.
    """
    I = img
    if I.ndim == 3:
        I = I[..., 0]
    H, W = I.shape[:2]
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    # Fast path via OpenCV remap when available; fallback to NumPy bilinear
    try:
        import cv2  # type: ignore
        map_x = u.astype(np.float32, copy=False)
        map_y = v.astype(np.float32, copy=False)
        remapped = cv2.remap(
            I.astype(np.float32, copy=False),
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        ).astype(float)
        # Mask out-of-bounds to NaN (remap used 0.0 at borders)
        mask = (u >= 0.0) & (u <= (W - 1)) & (v >= 0.0) & (v <= (H - 1))
        out = np.full(u.shape, np.nan, dtype=float)
        out[mask] = remapped[mask]
        return out
    except Exception:
        # NumPy bilinear (original semantics)
        u0 = np.floor(u).astype(int)
        v0 = np.floor(v).astype(int)
        u1 = u0 + 1
        v1 = v0 + 1
        du = u - u0
        dv = v - v0
        w00 = (1 - du) * (1 - dv)
        w10 = du * (1 - dv)
        w01 = (1 - du) * dv
        w11 = du * dv
        def inb(x, lo, hi):
            return (x >= lo) & (x < hi)
        m00 = inb(u0, 0, W) & inb(v0, 0, H)
        m10 = inb(u1, 0, W) & inb(v0, 0, H)
        m01 = inb(u0, 0, W) & inb(v1, 0, H)
        m11 = inb(u1, 0, W) & inb(v1, 0, H)
        out = np.full(u.shape, np.nan, dtype=float)
        acc = np.zeros(u.shape, dtype=float)
        val = np.zeros(u.shape, dtype=float)
        if m00.any():
            val[m00] += I[v0[m00], u0[m00]] * w00[m00]
            acc[m00] += w00[m00]
        if m10.any():
            val[m10] += I[v0[m10], u1[m10]] * w10[m10]
            acc[m10] += w10[m10]
        if m01.any():
            val[m01] += I[v1[m01], u0[m01]] * w01[m01]
            acc[m01] += w01[m01]
        if m11.any():
            val[m11] += I[v1[m11], u1[m11]] * w11[m11]
            acc[m11] += w11[m11]
        mask = acc > 0
        out[mask] = val[mask] / (acc[mask] + 1e-12)
        return out


def dense_flow(
    img0: np.ndarray, img1: np.ndarray, *, algo: str = "farneback"
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute dense optical flow between two grayscale images.

    Args:
        img0: Previous frame (H, W[,3]).
        img1: Current frame (H, W[,3]).
        algo: 'farneback' (default) or 'tvl1' if available.

    Returns:
        Tuple (fx, fy) dense flow fields (H, W) in pixels/frame.

    Raises:
        RuntimeError: If OpenCV is unavailable.
    """
    try:
        import cv2  # type: ignore
    except Exception as e:
        raise RuntimeError("OpenCV required for dense optical flow; install opencv-python") from e
    g0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY) if img0.ndim == 3 else img0
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if img1.ndim == 3 else img1
    if algo == "farneback_cuda":
        if not hasattr(cv2, "cuda"):
            raise RuntimeError("OpenCV was not built with CUDA (cv2.cuda missing). Install a CUDA-enabled OpenCV build or select --flow farneback.")
        try:
            devs = int(cv2.cuda.getCudaEnabledDeviceCount())
        except Exception:
            devs = 0
        if devs <= 0:
            raise RuntimeError("No CUDA devices visible to OpenCV (cv2.cuda.getCudaEnabledDeviceCount()==0). Check your CUDA install and drivers.")
        fb = cv2.cuda_FarnebackOpticalFlow.create(3, 0.5, False, 15, 3, 5, 1.2, 0)
        gpu0 = cv2.cuda_GpuMat(); gpu1 = cv2.cuda_GpuMat()
        gpu0.upload(g0.astype(np.uint8) if g0.dtype != np.uint8 else g0)
        gpu1.upload(g1.astype(np.uint8) if g1.dtype != np.uint8 else g1)
        flow_gpu = fb.calc(gpu0, gpu1, None)
        flow = flow_gpu.download()
    elif algo == "tvl1" and hasattr(cv2, "optflow"):
        tvl1 = cv2.optflow.DualTVL1OpticalFlow_create()
        flow = tvl1.calc(g0, g1, None)
    else:
        flow = cv2.calcOpticalFlowFarneback(
            g0,
            g1,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
    return flow[..., 0], flow[..., 1]


def compute_flow_stats_in_roi(
    img0: np.ndarray,
    img1: np.ndarray,
    center_uv: tuple[float, float],
    radius_px: float,
    *,
    algo: str = "farneback",
) -> Dict[str, float]:
    """Dense optical flow summary inside a circular ROI.

    Args:
        img0: Previous frame (H,W[,3]) uint8/float.
        img1: Current frame (H,W[,3]) uint8/float.
        center_uv: ROI center (u,v) in pixels.
        radius_px: ROI radius in pixels.
        algo: 'farneback' (default) or 'tvl1' if available.

    Returns:
        Dictionary with keys: flow_mag, flow_div, flow_curl, flow_dir_x, flow_dir_y.

    Raises:
        RuntimeError: If OpenCV is not available.
    """
    try:
        import cv2  # type: ignore
    except Exception as e:
        raise RuntimeError("OpenCV required for optical flow; install opencv-python") from e
    u0, v0 = center_uv
    H, W = img0.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W]
    mask = (xx - u0) ** 2 + (yy - v0) ** 2 <= float(radius_px) ** 2
    g0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY) if img0.ndim == 3 else img0
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if img1.ndim == 3 else img1
    if algo == "tvl1" and hasattr(cv2, "optflow"):
        tvl1 = cv2.optflow.DualTVL1OpticalFlow_create()
        flow = tvl1.calc(g0, g1, None)
    else:
        flow = cv2.calcOpticalFlowFarneback(
            g0,
            g1,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
    fx, fy = flow[..., 0], flow[..., 1]
    fxm, fym = fx[mask], fy[mask]
    if fxm.size == 0:
        return {
            "flow_mag": 0.0,
            "flow_div": 0.0,
            "flow_curl": 0.0,
            "flow_dir_x": 0.0,
            "flow_dir_y": 0.0,
        }
    mag = np.sqrt(fxm * fxm + fym * fym)
    dudx = np.gradient(fx, axis=1)
    dvdy = np.gradient(fy, axis=0)
    dvdx = np.gradient(fy, axis=1)
    dudy = np.gradient(fx, axis=0)
    div = (dudx + dvdy)[mask]
    curl = (dvdx - dudy)[mask]
    mx, my = float(np.mean(fxm)), float(np.mean(fym))
    norm = math.sqrt(mx * mx + my * my) + 1e-12
    return {
        "flow_mag": float(np.mean(mag)),
        "flow_div": float(np.mean(div)),
        "flow_curl": float(np.mean(curl)),
        "flow_dir_x": mx / norm,
        "flow_dir_y": my / norm,
    }


def compute_motion_energy_in_roi(
    img: np.ndarray,
    center_uv: tuple[float, float],
    radius_px: float,
    *,
    bank: Optional[Sequence[np.ndarray]] = None,
) -> np.ndarray:
    """Motion-energy (Gabor squared-energy) summary inside a circular ROI.

    Args:
        img: Frame image (H,W[,3]).
        center_uv: ROI center (u,v) in pixels.
        radius_px: ROI radius in pixels.
        bank: Optional iterable of 2D Gabor kernels; default creates a small bank.

    Returns:
        1D array of mean squared-energy per filter.

    Raises:
        RuntimeError: If OpenCV is not available.
    """
    try:
        import cv2  # type: ignore
    except Exception as e:
        raise RuntimeError("OpenCV required for motion-energy; install opencv-python") from e
    u0, v0 = center_uv
    H, W = img.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W]
    mask = (xx - u0) ** 2 + (yy - v0) ** 2 <= float(radius_px) ** 2
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    kernels = list(bank) if bank is not None else []
    if not kernels:
        for th in (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4):
            for lam in (4.0, 8.0):
                k = cv2.getGaborKernel(ksize=(9, 9), sigma=2.0, theta=th, lambd=lam, gamma=0.5, psi=0)
                kernels.append(k)
    feats = []
    for k in kernels:
        r = cv2.filter2D(g, cv2.CV_32F, k)
        v = r * r
        feats.append(float(np.mean(v[mask])))
    return np.asarray(feats, dtype=float)


def pca_topk(X: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (scores, components) for top-k PCA using SVD.

    Args:
        X: (N,D) matrix.
        k: Number of principal components.

    Returns:
        scores: (N,k) matrix of principal component scores.
        components: (k,D) matrix of top-k right singular vectors.
    """
    X = np.asarray(X, dtype=float)
    Xc = X - np.nanmean(X, axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(np.nan_to_num(Xc), full_matrices=False)
    kk = int(min(max(k, 1), Vt.shape[0]))
    return Xc @ Vt[:kk].T, Vt[:kk]
