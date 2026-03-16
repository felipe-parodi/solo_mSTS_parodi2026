"""Spikes loaders for NPZ and MAT formats with event-coded output.

Self-contained: does not depend on external repos. For MAT, uses scipy.io to
read either Unit_rasters (preferred) or a minimal SpikeData fallback.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..types import SpikesEvents, SessionRef

# Precompiled regex for fast digit extraction from channel/unit names
_DIGITS_RE = re.compile(r'\d+')


# ---------------------------------------------------------------------------
# Low-level MAT parsing helpers (hoisted from load_spikes_events_mat internals)
# ---------------------------------------------------------------------------


def _uf_sort_key(uf: str) -> Tuple[int, str]:
    """Sort key for unit field names — numeric suffix first, then name."""
    try:
        m = _DIGITS_RE.search(str(uf))
        return (int(m.group()) if m else 0, str(uf))
    except (ValueError, TypeError):
        return (1 << 30, str(uf))


def _iter_channel_names_scipy(sd_obj: Any) -> List[str]:
    """Extract channel field names from a scipy mat_struct or structured array."""
    names: Any = getattr(sd_obj, 'dtype', type('x', (), {'names': []})).names or []
    if (not names) and hasattr(sd_obj, '_fieldnames') and getattr(sd_obj, '_fieldnames'):
        try:
            names = list(sd_obj._fieldnames)  # type: ignore[attr-defined]
        except (TypeError, AttributeError):
            names = []
    out: List[str] = []
    for nm in names or []:
        if not isinstance(nm, str):
            continue
        ln = nm.lower()
        if ln.startswith('channel_') or ln.startswith('channel') or ln.startswith('ch'):
            out.append(nm)
    return out


def _peel_cell1(x: Any) -> Any:
    """Unwrap single-element MATLAB cell arrays (object arrays of size 1)."""
    if isinstance(x, np.ndarray) and x.dtype == object and x.size == 1:
        return x.flat[0]
    if isinstance(x, np.ndarray) and x.dtype == object and x.shape == (1, 1):
        return x[0, 0]
    return x


def _as_1d_float(x: Any) -> np.ndarray:
    """Convert value to 1D float array, returning empty on failure."""
    try:
        arr = np.asarray(x)
        if arr.size == 0:
            return np.asarray([], dtype=float)
        arr = arr.astype(float).reshape(-1)
        return arr
    except (ValueError, TypeError):
        return np.asarray([], dtype=float)


def _get_unit_fields(obj: Any) -> List[str]:
    """Extract and sort unit field names from a channel struct."""
    unit_fields: Any = getattr(obj, 'dtype', type('x', (), {'names': []})).names or []
    if (not unit_fields) and hasattr(obj, '_fieldnames') and getattr(obj, '_fieldnames'):
        try:
            unit_fields = list(obj._fieldnames)  # type: ignore[attr-defined]
        except (TypeError, AttributeError):
            unit_fields = []
    if not unit_fields:
        return []
    ufs: List[str] = []
    for uf in unit_fields:
        if isinstance(uf, str) and uf.lower().startswith('unit'):
            ufs.append(uf)
    return sorted(ufs, key=_uf_sort_key) if ufs else list(unit_fields)


def _resolve_channel_container(sd_obj: Any, name: str) -> Any:
    """Resolve a channel container from a SpikeData struct, unwrapping (1,1) cells."""
    chan_raw = getattr(sd_obj, name, None)
    if chan_raw is None:
        try:
            chan_raw = sd_obj[name]  # type: ignore[index]
        except (KeyError, IndexError, TypeError):
            chan_raw = None
    if isinstance(chan_raw, np.ndarray) and chan_raw.dtype == object and chan_raw.shape == (1, 1):
        chan_raw = chan_raw[0, 0]
    return chan_raw


def _extract_units_for_channel(chan_raw: Any) -> List[Tuple[np.ndarray, str]]:
    """Extract (times, kind) pairs for all units in one channel container.

    Handles three SpikeData representations:
      A. Object array of unit vectors
      B. Struct with named unit fields
      C. Bare numeric vector (single MUA)
    """
    out_units: List[Tuple[np.ndarray, str]] = []
    # Case A: object array of unit vectors
    if isinstance(chan_raw, np.ndarray) and chan_raw.dtype == object:
        n_units = int(chan_raw.size)
        for j in range(n_units):
            u_obj = chan_raw.flat[j]
            u_obj = _peel_cell1(u_obj)
            st = _as_1d_float(u_obj)
            if st.size:
                kind = 'MUA' if j == 0 else 'SUA'
                out_units.append((st, kind))
        return out_units
    # Case B: struct with unit fields
    ufs = _get_unit_fields(chan_raw)
    if ufs:
        for j, uf in enumerate(ufs):
            u = getattr(chan_raw, uf, None)
            if u is None:
                try:
                    u = chan_raw[uf]  # type: ignore[index]
                except (KeyError, IndexError, TypeError):
                    u = None
            u = _peel_cell1(u)
            st = _as_1d_float(u)
            if st.size:
                kind = 'MUA' if j == 0 else 'SUA'
                out_units.append((st, kind))
        return out_units
    # Case C: bare numeric vector → single MUA
    st = _as_1d_float(chan_raw)
    if st.size:
        out_units.append((st, 'MUA'))
    return out_units


def _enumerate_unit_kinds(chan_raw: Any) -> List[str]:
    """Count and classify units without loading spike times.

    Used for metadata augmentation of Unit_rasters data.
    """
    # Case A: object array
    if isinstance(chan_raw, np.ndarray) and chan_raw.dtype == object:
        return ['MUA' if j == 0 else 'SUA' for j in range(int(chan_raw.size))]
    # Case B: struct with unit fields
    ufs = _get_unit_fields(chan_raw)
    if ufs:
        return ['MUA' if j == 0 else 'SUA' for j in range(len(ufs))]
    # Case C: bare numeric vector
    return ['MUA']


def _channel_index_from_name(name: str) -> int:
    """Extract 0-based channel index from a 1-based MATLAB channel name."""
    try:
        m = _DIGITS_RE.search(str(name))
        nn = int(m.group()) if m else 0
    except (ValueError, TypeError):
        nn = 0
    return max(nn - 1, 0)


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def _compute_cache_key(mat_path: str) -> Tuple[Path, str]:
    """Compute cache key from MAT file path, size, and mtime."""
    p = Path(mat_path)
    st = p.stat()
    sig = f"{str(p.resolve())}|{st.st_size}|{st.st_mtime_ns}"
    key = hashlib.md5(sig.encode("utf-8")).hexdigest()[:16]
    return p, key


def _cache_write_npz(
    mat_path: str,
    cache_dir: Optional[str],
    data: Dict[str, Any],
    event_source: str,
    bin_width_s: float,
    *,
    verbose: bool = False,
) -> None:
    """Write spike data to NPZ cache file.

    Args:
        mat_path: Source MAT file path (used for cache key).
        cache_dir: Cache directory path, or None to skip.
        data: Spike data dict with spike_t, spike_unit_index, etc.
        event_source: Source identifier ('SpikeData' or 'Unit_rasters').
        bin_width_s: Bin width in seconds (NaN for SpikeData).
        verbose: Print cache write messages.
    """
    if cache_dir is None:
        return
    try:
        p, key = _compute_cache_key(mat_path)
        cache_base = Path(cache_dir)
        cache_base.mkdir(parents=True, exist_ok=True)
        npz_path = cache_base / f"{p.stem}.{key}.events.npz"
        np.savez_compressed(
            str(npz_path),
            unit_ids=data.get("unit_ids", np.array([], dtype=object)),
            spike_t=data["spike_t"],
            spike_unit_index=data["spike_unit_index"],
            session_id=data.get("session_id", p.stem),
            unit_channels=data.get("unit_channels", np.array([], dtype=np.int32)),
            unit_kind=data.get("unit_kind", np.array([], dtype=object)),
            event_source=np.asarray(event_source, dtype=object),
            bin_width_s=np.asarray(bin_width_s, dtype=float),
        )
        if verbose:
            print(f"[spikes] Cached to {npz_path.name}")
    except Exception as e:
        if verbose:
            print(f"[spikes] Cache write failed: {e}")


# ---------------------------------------------------------------------------
# HDF5 loader (unchanged from original — already clean)
# ---------------------------------------------------------------------------


def _is_hdf5_mat(path: str) -> bool:
    """Check if a MAT file is HDF5-based (v7.3 format)."""
    try:
        with open(path, "rb") as f:
            header = f.read(8)
        return header == b"\x89HDF\r\n\x1a\n"
    except (OSError, IOError):
        return False


def _load_spikedata_h5(path: str) -> Optional[Dict[str, Any]]:
    """Load SpikeData directly from HDF5-based MAT file (v7.3 format).

    Returns None if SpikeData not found or loading fails.
    Raises ImportError if h5py not available.
    """
    try:
        import h5py
    except ImportError as e:
        raise ImportError("h5py is required for v7.3 MAT files. Install with: pip install h5py") from e

    try:
        print(f"[h5py] Loading SpikeData from HDF5 MAT: {os.path.basename(path)}")
        with h5py.File(path, 'r') as f:
            # Find SpikeData group
            sd = None
            if 'SpikeData' in f:
                sd = f['SpikeData']
            elif '/SpikeData' in f:
                sd = f['/SpikeData']

            if sd is None or not isinstance(sd, h5py.Group):
                print("[h5py] No SpikeData group found, falling back to scipy")
                return None

            # Extract channel data
            unit_t_list: List[np.ndarray] = []
            unit_channel: List[int] = []
            unit_kind: List[str] = []

            # Iterate over channel groups
            channel_names = [k for k in sd.keys() if k.lower().startswith(('channel', 'ch'))]
            if not channel_names:
                print("[h5py] No channel groups found in SpikeData")
                return None

            print(f"[h5py] Found {len(channel_names)} channels")

            for chan_name in sorted(channel_names):
                # Extract channel index from name (1-based MATLAB → 0-based Python)
                m = _DIGITS_RE.search(chan_name)
                ch_idx = (int(m.group()) - 1) if m else 0

                chan_grp = sd[chan_name]

                # Extract units from this channel
                if isinstance(chan_grp, h5py.Group):
                    # Case A: Group with unit fields
                    unit_fields = [k for k in chan_grp.keys() if 'unit' in k.lower()]
                    if unit_fields:
                        # Sort unit fields by numeric suffix
                        def _uf_key(uf: str) -> Tuple[int, str]:
                            m = _DIGITS_RE.search(uf)
                            return (int(m.group()) if m else 0, uf)
                        unit_fields = sorted(unit_fields, key=_uf_key)

                        for j, uf in enumerate(unit_fields):
                            unit_data = chan_grp[uf]
                            if isinstance(unit_data, h5py.Dataset):
                                times = np.asarray(unit_data).flatten().astype(float)
                                if times.size > 0:
                                    unit_t_list.append(times)
                                    unit_channel.append(ch_idx)
                                    unit_kind.append('MUA' if j == 0 else 'SUA')
                    else:
                        # Single dataset per channel
                        if len(chan_grp.keys()) == 1:
                            key = list(chan_grp.keys())[0]
                            unit_data = chan_grp[key]
                            if isinstance(unit_data, h5py.Dataset):
                                times = np.asarray(unit_data).flatten().astype(float)
                                if times.size > 0:
                                    unit_t_list.append(times)
                                    unit_channel.append(ch_idx)
                                    unit_kind.append('MUA')
                elif isinstance(chan_grp, h5py.Dataset):
                    # Case B: Direct dataset (rare but possible)
                    times = np.asarray(chan_grp).flatten().astype(float)
                    if times.size > 0:
                        unit_t_list.append(times)
                        unit_channel.append(ch_idx)
                        unit_kind.append('MUA')

            if not unit_t_list:
                print("[h5py] No spike data extracted from channels")
                return None

            # Build event-coded arrays
            U = len(unit_t_list)
            print(f"[h5py] Extracted {U} units, building event arrays...")

            spike_t = np.concatenate([
                st.astype(np.float32, copy=False) for st in unit_t_list if st.size > 0
            ])
            spike_unit_index = np.concatenate([
                np.full(st.size, ui, dtype=np.int32)
                for ui, st in enumerate(unit_t_list) if st.size > 0
            ])

            # Sort by time
            order = np.argsort(spike_t, kind='quicksort')
            spike_t = spike_t[order]
            spike_unit_index = spike_unit_index[order]

            unit_ids = np.array([f"ch{unit_channel[i]:03d}_{unit_kind[i]}" for i in range(U)], dtype=object)

            print(f"[h5py] Success: {spike_t.size} events from {U} units")

            return {
                "unit_ids": unit_ids,
                "spike_t": spike_t,
                "spike_unit_index": spike_unit_index,
                "session_id": os.path.splitext(os.path.basename(path))[0],
                "unit_channels": np.asarray(unit_channel, dtype=np.int32),
                "unit_kind": np.asarray(unit_kind, dtype=object),
                "event_source": "SpikeData",
                "bin_width_s": None,
            }

    except ImportError:
        raise
    except Exception as e:
        print(f"[h5py] Failed to load SpikeData: {e}")
        print("[h5py] Falling back to scipy.io.loadmat")
        return None


# ---------------------------------------------------------------------------
# SpikeData extraction (scipy path)
# ---------------------------------------------------------------------------


def _extract_spikedata_scipy(sd: Any, mat_path: str) -> Optional[Dict[str, Any]]:
    """Extract event-coded spike arrays from a scipy-loaded SpikeData struct.

    Args:
        sd: SpikeData struct from scipy.io.loadmat.
        mat_path: MAT file path (for session_id).

    Returns:
        Spike data dict, or None if no channels found.
    """
    unit_t_list: List[np.ndarray] = []
    unit_channel: List[int] = []
    unit_kind: List[str] = []

    for name in _iter_channel_names_scipy(sd):
        ch_idx0 = _channel_index_from_name(name)
        chan_raw = _resolve_channel_container(sd, name)
        for st, kind in _extract_units_for_channel(chan_raw):
            unit_t_list.append(st)
            unit_channel.append(ch_idx0)
            unit_kind.append(kind)

    if not unit_t_list:
        return None

    # Build event-coded arrays with vectorized operations
    U = len(unit_t_list)
    spike_t = np.concatenate([
        st.astype(np.float32, copy=False) for st in unit_t_list if st.size > 0
    ])
    spike_unit_index = np.concatenate([
        np.full(st.size, ui, dtype=np.int32)
        for ui, st in enumerate(unit_t_list) if st.size > 0
    ])
    # Sort events by time (quicksort faster than mergesort for this use case)
    order = np.argsort(spike_t, kind='quicksort')
    spike_t = spike_t[order]
    spike_unit_index = spike_unit_index[order]
    unit_ids = np.array([f"ch{unit_channel[i]:03d}_{unit_kind[i]}" for i in range(U)], dtype=object)

    return {
        "unit_ids": unit_ids,
        "spike_t": spike_t,
        "spike_unit_index": spike_unit_index,
        "session_id": os.path.splitext(os.path.basename(mat_path))[0],
        "unit_channels": np.asarray(unit_channel, dtype=np.int32),
        "unit_kind": np.asarray(unit_kind, dtype=object),
        "event_source": "SpikeData",
        "bin_width_s": None,
    }


# ---------------------------------------------------------------------------
# Unit_rasters extraction
# ---------------------------------------------------------------------------


def _extract_unit_rasters_events(mat: dict, mat_path: str) -> Dict[str, Any]:
    """Convert Unit_rasters count matrix to event-coded arrays.

    Args:
        mat: Loaded MAT dict from scipy.io.loadmat.
        mat_path: MAT file path (for session_id).

    Returns:
        Spike data dict with event_source='Unit_rasters'.

    Raises:
        ValueError: If Unit_rasters not found or has wrong dimensions.
    """
    if 'Unit_rasters' not in mat:
        raise ValueError(f"No usable SpikeData or Unit_rasters found in {mat_path}")
    ur = np.asarray(mat['Unit_rasters'])
    if ur.ndim != 2:
        raise ValueError(f"No usable SpikeData or Unit_rasters found in {mat_path}")
    if ur.shape[0] <= ur.shape[1]:
        counts = ur.T.astype(np.int64)
    else:
        counts = ur.astype(np.int64)

    n_units = counts.shape[1]
    bin_used = 1.0  # Unit_rasters always uses 1s bins
    t = (np.arange(counts.shape[0], dtype=np.float32) + 0.5) * bin_used
    tt, uu = np.nonzero(counts)
    reps = counts[tt, uu].astype(np.int64)
    spike_t = np.repeat(t[tt], reps).astype(np.float32, copy=False)
    spike_unit_index = np.repeat(uu.astype(np.int32), reps).astype(np.int32, copy=False)
    unit_ids = np.array([f"unit_{i}" for i in range(n_units)], dtype=object)

    return {
        "unit_ids": unit_ids,
        "spike_t": spike_t,
        "spike_unit_index": spike_unit_index,
        "session_id": os.path.splitext(os.path.basename(mat_path))[0],
        "event_source": "Unit_rasters",
        "bin_width_s": float(bin_used),
        "_n_units": n_units,
    }


def _augment_rasters_with_spikedata_meta(
    d_final: Dict[str, Any],
    mat: dict,
    mat_path: str,
    n_units: int,
    whosmat_names: Optional[set],
) -> None:
    """Attempt to add unit_channels/unit_kind from SpikeData to Unit_rasters output.

    Tries scipy struct inspection first, falls back to h5py probe for v7.3 MAT.
    Attaches metadata only if the unit count matches exactly.

    Args:
        d_final: Unit_rasters output dict (modified in place).
        mat: Loaded MAT dict.
        mat_path: MAT file path.
        n_units: Number of units in Unit_rasters.
        whosmat_names: Set of variable names from whosmat, or None if unavailable.
    """
    try:
        sd2 = mat.get('SpikeData')
        unit_channel_meta: List[int] = []
        unit_kind_meta: List[str] = []

        if sd2 is not None:
            chan_names = _iter_channel_names_scipy(sd2)
            for name in chan_names:
                chan_struct = _resolve_channel_container(sd2, name)
                kinds_here = _enumerate_unit_kinds(chan_struct)
                ch0 = _channel_index_from_name(name)
                for k in kinds_here:
                    unit_channel_meta.append(int(ch0))
                    unit_kind_meta.append(k)

        # If SciPy view failed (no names), try a light HDF5 probe for v7.3 MAT
        names = getattr(sd2, 'dtype', type('x', (), {'names': []})).names or [] if sd2 is not None else []
        if (not unit_channel_meta) and (sd2 is None or not names):
            _augment_via_h5py_probe(unit_channel_meta, unit_kind_meta, mat_path)

        # If sizes match, attach metadata
        if unit_channel_meta and len(unit_channel_meta) == n_units:
            d_final["unit_channels"] = np.asarray(unit_channel_meta, dtype=np.int32)
            d_final["unit_kind"] = np.asarray(unit_kind_meta, dtype=object)
    except Exception:
        pass


def _augment_via_h5py_probe(
    unit_channel_meta: List[int],
    unit_kind_meta: List[str],
    mat_path: str,
) -> None:
    """Probe HDF5 MAT file structure to infer unit channel/kind metadata.

    Modifies unit_channel_meta and unit_kind_meta in place.
    """
    try:
        import h5py as _h5
        with _h5.File(str(mat_path), 'r') as _f:
            cand_paths = []
            if 'SpikeData' in _f:
                cand_paths.append('SpikeData')
            if '/SpikeData' in _f:
                cand_paths.append('/SpikeData')
            for cp in cand_paths:
                try:
                    node = _f[cp]
                    if not isinstance(node, _h5.Group):
                        continue
                    for nm in list(node.keys()):
                        ln = str(nm).lower()
                        if not (ln.startswith('channel') or ln.startswith('ch')):
                            continue
                        ch_idx = _channel_index_from_name(str(nm))
                        g = node[nm]
                        n_units_here = 0
                        if isinstance(g, _h5.Group):
                            unit_like = [k for k in g.keys() if 'unit' in str(k).lower()]
                            n_units_here = len(unit_like) if unit_like else 1
                        else:
                            n_units_here = 1
                        for j in range(int(n_units_here)):
                            unit_channel_meta.append(int(ch_idx))
                            unit_kind_meta.append('MUA' if j == 0 else ('SUA' if j == 1 else 'U2+'))
                    # Stop after first usable path
                    if unit_channel_meta:
                        break
                except Exception:
                    continue
    except Exception:
        pass


# ---------------------------------------------------------------------------
# NPZ loaders
# ---------------------------------------------------------------------------


def _load_spikes_npz(path: str) -> Dict[str, Any]:
    with np.load(path, allow_pickle=False) as z:
        files = set(z.files)
        if {"spike_t", "spike_unit_index"}.issubset(files):
            unit_ids = z["unit_ids"] if "unit_ids" in files else np.array([], dtype=object)
            return {
                "unit_ids": unit_ids,
                "spike_t": z["spike_t"].astype(np.float32),
                "spike_unit_index": z["spike_unit_index"].astype(np.int32),
                "session_id": (z["session_id"].item() if "session_id" in files else os.path.splitext(os.path.basename(path))[0]),
            }
        # fallback from binned to events at bin centers
        if {"binned_spikes", "timestamps"}.issubset(files):
            counts = z["binned_spikes"]
            timestamps = z["timestamps"].astype(np.float32)
            if counts.ndim != 2 or timestamps.ndim != 1 or timestamps.shape[0] != counts.shape[0]:
                raise ValueError("Invalid binned_spikes/timestamps shapes")
            tt, uu = np.nonzero(counts)
            reps = counts[tt, uu].astype(np.int64)
            spike_t = np.repeat(timestamps[tt], reps).astype(np.float32, copy=False)
            spike_unit_index = np.repeat(uu.astype(np.int32), reps).astype(np.int32, copy=False)
            unit_ids = z["unit_ids"] if "unit_ids" in files else np.array([f"unit_{i}" for i in range(counts.shape[1])], dtype=object)
            return {
                "unit_ids": unit_ids,
                "spike_t": spike_t,
                "spike_unit_index": spike_unit_index,
                "session_id": (z["session_id"].item() if "session_id" in files else os.path.splitext(os.path.basename(path))[0]),
            }
        raise ValueError("Unsupported spikes NPZ schema (need spike_t/spike_unit_index or binned_spikes/timestamps)")


def load_spikes_events_npz(npz_path: str) -> SpikesEvents:
    """Load spikes from NPZ file."""
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(npz_path)
    d = _load_spikes_npz(npz_path)
    return SpikesEvents(
        t=d["spike_t"],
        unit_index=d["spike_unit_index"],
        # Avoid boolean evaluation on numpy arrays; handle None/missing explicitly
        unit_ids=(
            np.asarray(d["unit_ids"]) if ("unit_ids" in d and d["unit_ids"] is not None)
            else np.asarray([], dtype=object)
        ),
        session_id=str(d.get("session_id") or os.path.splitext(os.path.basename(npz_path))[0]),
        meta={k: d[k] for k in ("unit_channels", "unit_kind") if k in d},
    )


def load_spikes_from_session(session: SessionRef) -> Optional[SpikesEvents]:
    """Load spikes by inspecting session.data_files for *_neural or spikes_npz keys."""
    # Prefer explicit NPZ if present
    spikes_npz = session.data_files.get("spikes_npz")
    if spikes_npz and os.path.isfile(str(spikes_npz)):
        return load_spikes_events_npz(str(spikes_npz))

    # Try known neural MAT keys from config
    for k, v in session.data_files.items():
        if not k.endswith("_neural") or not v:
            continue
        path = str(v)
        if path.lower().endswith(".npz"):
            return load_spikes_events_npz(path)
        if path.lower().endswith(".mat"):
            d = load_spikes_events_mat(path)
            return SpikesEvents(
                t=d["spike_t"],
                unit_index=d["spike_unit_index"],
                unit_ids=(
                    np.asarray(d["unit_ids"]) if ("unit_ids" in d and d["unit_ids"] is not None)
                    else np.asarray([], dtype=object)
                ),
                session_id=str(d.get("session_id") or os.path.splitext(os.path.basename(path))[0]),
                meta={k: d[k] for k in ("unit_channels", "unit_kind", "event_source", "bin_width_s") if k in d},
            )
    return None


# ---------------------------------------------------------------------------
# Main MAT loader — thin router with cache
# ---------------------------------------------------------------------------


def load_spikes_events_mat(
    mat_path: str,
    *,
    bin_size_s: float = 1.0,
    use_cache: bool = True,
    cache_dir: Optional[str] = "cache/spikes",
    prefer_spikedata: bool = True,
) -> Dict[str, Any]:
    """Load spikes from a FreeMat .mat file and return event-coded arrays.

    Strategy:
      - Prefer SpikeData (per-channel units) to extract event timestamps and unit channels.
      - Fallback to 'Unit_rasters' (units x time or time x units) and synthesize events.
    """
    try:
        import scipy.io  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise ImportError("scipy is required to load .mat spikes") from e

    # Inspect variables to limit load to what's needed
    var_names: Optional[List[str]] = None
    whosmat_names: Optional[set] = None
    try:
        wh = scipy.io.whosmat(mat_path)
        whosmat_names = {w[0] for w in wh}
        wanted: List[str] = []
        if "SpikeData" in whosmat_names:
            wanted.append("SpikeData")
        if "Unit_rasters" in whosmat_names:
            wanted.append("Unit_rasters")
        if wanted:
            var_names = wanted
    except Exception:
        var_names = None

    # --- Cache read ---
    cache_hit = False
    if use_cache:
        result = _try_cache_read(
            mat_path, cache_dir, var_names, prefer_spikedata,
        )
        if result is not None:
            return result

    # --- Try h5py direct loading for HDF5 MAT files (v7.3) - 3-5x faster ---
    if not cache_hit and prefer_spikedata:
        is_hdf5 = _is_hdf5_mat(mat_path)
        if is_hdf5:
            print("[spikes] Detected HDF5 MAT file (v7.3 format)")
            h5_data = _load_spikedata_h5(mat_path)
            if h5_data is not None:
                if use_cache:
                    _cache_write_npz(
                        mat_path, cache_dir, h5_data,
                        event_source="SpikeData", bin_width_s=float('nan'),
                        verbose=True,
                    )
                return h5_data
        else:
            print("[spikes] Detected scipy-compatible MAT file (v7 or earlier)")

    # --- Load with scipy.io.loadmat ---
    print("[scipy] Loading MAT file with scipy.io.loadmat...")
    try:
        mat = scipy.io.loadmat(
            mat_path,
            variable_names=var_names,
            squeeze_me=True,
            struct_as_record=False,
        )
    except TypeError:
        # Older SciPy without variable_names support
        mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)

    # --- Try SpikeData extraction ---
    sd = mat.get('SpikeData')
    if sd is not None:
        d_sd = _extract_spikedata_scipy(sd, mat_path)
        if d_sd is not None:
            if use_cache:
                _cache_write_npz(
                    mat_path, cache_dir, d_sd,
                    event_source="SpikeData", bin_width_s=float('nan'),
                )
            return d_sd

    # --- Fallback: Unit_rasters → events at bin centers ---
    d_final = _extract_unit_rasters_events(mat, mat_path)
    n_units = d_final.pop("_n_units")

    # Attempt to augment with channel/kind metadata from SpikeData
    _augment_rasters_with_spikedata_meta(d_final, mat, mat_path, n_units, whosmat_names)

    # Save cache
    if use_cache:
        _cache_write_npz(
            mat_path, cache_dir, d_final,
            event_source="Unit_rasters", bin_width_s=d_final.get("bin_width_s", 1.0),
        )

    return d_final


def _try_cache_read(
    mat_path: str,
    cache_dir: Optional[str],
    var_names: Optional[List[str]],
    prefer_spikedata: bool,
) -> Optional[Dict[str, Any]]:
    """Attempt to read spike data from NPZ cache.

    Returns cached data dict, or None if cache miss or stale.
    """
    try:
        p, key = _compute_cache_key(mat_path)
        cache_base = Path(cache_dir) if cache_dir is not None else None
        if cache_base is None:
            return None
        npz_path = cache_base / f"{p.stem}.{key}.events.npz"
        if not npz_path.exists():
            return None

        # allow_pickle=True is safe here - we control these cache files
        with np.load(str(npz_path), allow_pickle=True) as z:
            files = set(z.files)
            if not {"spike_t", "spike_unit_index"}.issubset(files):
                return None

            # Check event source to avoid using Unit_rasters cache when SpikeData is available
            src = None
            if "event_source" in files:
                try:
                    src = str(z["event_source"].item())
                except (ValueError, TypeError):
                    src = None

            # Robustly determine if the MAT likely contains SpikeData
            has_spikedata = False
            if var_names is not None and "SpikeData" in var_names:
                has_spikedata = True
            elif var_names is None:
                # Only probe HDF5 as fallback when var_names unavailable (rare)
                try:
                    with open(str(p), "rb") as _fh:
                        hdr = _fh.read(8)
                    is_hdf5 = hdr == b"\x89HDF\r\n\x1a\n"
                    if is_hdf5:
                        import h5py as _h5
                        with _h5.File(str(p), "r") as _f:
                            has_spikedata = "SpikeData" in _f
                except Exception:
                    has_spikedata = False

            # Check if cache needs rebuild
            U_cached = None
            try:
                if "spike_unit_index" in files:
                    U_cached = int(np.max(z["spike_unit_index"])) + 1
            except (ValueError, TypeError):
                U_cached = None
            needs_meta = (
                ("unit_channels" not in files)
                or ("unit_channels" in files and z["unit_channels"].size == 0)
                or ("unit_channels" in files and U_cached is not None and z["unit_channels"].size not in (0, U_cached))
            )
            prefer_rebuild = bool(
                (prefer_spikedata and has_spikedata and (src in (None, "Unit_rasters")))
                or (prefer_spikedata and needs_meta)
            )
            if prefer_rebuild:
                return None

            # Build output from cache
            d_out: Dict[str, Any] = {
                "unit_ids": z["unit_ids"],
                "spike_t": z["spike_t"],
                "spike_unit_index": z["spike_unit_index"],
                "session_id": z["session_id"].item() if "session_id" in files else p.stem,
            }
            if "unit_channels" in files:
                d_out["unit_channels"] = z["unit_channels"]
            if "unit_kind" in files:
                d_out["unit_kind"] = z["unit_kind"]
            # Carry event_source when present
            inferred_src = None
            if "event_source" in files:
                try:
                    inferred_src = str(z["event_source"].item())
                except (ValueError, TypeError):
                    inferred_src = None
            if "bin_width_s" in files:
                try:
                    bw = float(z["bin_width_s"])
                    if inferred_src is None:
                        inferred_src = "Unit_rasters" if np.isfinite(bw) else "SpikeData"
                    d_out["bin_width_s"] = bw
                except (ValueError, TypeError):
                    pass
            if inferred_src is not None:
                d_out["event_source"] = inferred_src
            print(f"[cache] Loaded from cache: {npz_path.name}")
            return d_out
    except Exception:
        return None
