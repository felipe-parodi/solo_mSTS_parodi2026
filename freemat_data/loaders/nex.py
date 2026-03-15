from __future__ import annotations

"""NeuroExplorer (.nex/.nex5) continuous loader.

This module provides a minimal, typed loader that returns LFP-like continuous
data from NeuroExplorer files as an ``LFPTimeSeries``. It favors a simple public
API and fails loudly with clear errors.

Dependencies:
    - neo >= 0.12 (``pip install neo``)

Notes:
    - This loader uses Neo's NeuroExplorerRawIO and reads only analog signals
      from the first block/segment. Use the ``--duration`` option in the QC
      CLI to limit read length for long recordings.
    - Values are returned as plain floats with no unit scaling. Inspect the
      resulting magnitude to determine if values are V or uV and rescale
      externally if needed for plotting aesthetics.

Example:
    >>> from freemat_data.loaders.nex import load_continuous_from_nex
    >>> lfp = load_continuous_from_nex("/path/to/file.nex5")
    >>> lfp.fs, lfp.x.shape
    (30000.0, (T, C))
"""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from ..types import LFPTimeSeries


def read_nex_header(filepath: str) -> dict:
    """Read and summarize header for a .nex/.nex5 file via RawIO.

    Returns a plain dict with 'streams' and 'channels' entries to help
    choose valid channels/streams and their sampling rates.

    Raises FileNotFoundError or RuntimeError on missing keys.
    """
    p = Path(filepath)
    if not p.exists():
        raise FileNotFoundError(f"read_nex_header: path not found: {filepath}")

    from neo.rawio import NeuroExplorerRawIO  # type: ignore
    import neo.rawio.neuroexplorerrawio as nraw  # type: ignore
    import inspect
    import numpy as _np

    # Ensure tolerant decoding as in the main loader
    src = inspect.getsource(nraw.read_as_dict)
    if any(s in src for s in ("decode(\"utf8\")", "decode('utf8')", "decode(\"utf-8\")", "decode('utf-8')")):
        patched = (
            src.replace("decode(\"utf8\")", "decode('latin-1', errors='replace')")
            .replace("decode('utf8')", "decode('latin-1', errors='replace')")
            .replace("decode(\"utf-8\")", "decode('latin-1', errors='replace')")
            .replace("decode('utf-8')", "decode('latin-1', errors='replace')")
        )
        exec(patched, nraw.__dict__)

    reader = NeuroExplorerRawIO(filename=str(p))
    reader.parse_header()
    header = getattr(reader, "header", None)
    if header is None:
        raise RuntimeError("read_nex_header: reader.header is None")

    out: dict = {"file": str(p)}

    # Streams
    streams = []
    if "signal_streams" in header:
        ss = header["signal_streams"]
        for i in range(int(ss.size)):
            sr = float(ss["sampling_rate"][i]) if "sampling_rate" in ss.dtype.names else float("nan")
            streams.append(
                {
                    "index": i,
                    "id": str(ss["id"][i]) if "id" in ss.dtype.names else None,
                    "name": str(ss["name"][i]) if "name" in ss.dtype.names else None,
                    "sampling_rate": sr,
                }
            )
    out["streams"] = streams

    # Channels
    if "signal_channels" not in header:
        raise RuntimeError("read_nex_header: header missing 'signal_channels'")
    sc = header["signal_channels"]
    chans = []
    names = sc.dtype.names
    for i in range(int(sc.size)):
        rec = sc[i]
        entry = {"index": i}
        for key in ("id", "name", "channel_id", "stream_id", "units", "dtype"):
            if key in names:
                entry[key] = None if rec[key] is None else str(rec[key])
        # Per-channel sampling rate when present
        if "sampling_rate" in names:
            try:
                entry["sampling_rate"] = float(rec["sampling_rate"])  # type: ignore[arg-type]
            except Exception:
                entry["sampling_rate"] = float("nan")
        else:
            entry["sampling_rate"] = float("nan")
        chans.append(entry)
    out["channels"] = chans

    # If RawIO reports no or 1 channel and the file is HDF5, enrich with HDF view
    if len(chans) <= 1 and _is_hdf5_file(str(p)):
        h5 = _nex5_hdf5_summary(str(p))
        out["hdf5"] = h5

    return out


def _is_hdf5_file(path: str) -> bool:
    with open(path, "rb") as f:
        sig = f.read(8)
    return sig == b"\x89HDF\r\n\x1a\n"


def _h5_get_fs_from_group(g) -> Optional[float]:
    # Attrs
    for key in ("SamplingRate", "WFrequency", "SamplingFrequency", "Frequency", "Fs"):
        if key in g.attrs:
            try:
                return float(g.attrs[key])
            except Exception:
                pass
    # Datasets
    for key in ("SamplingRate", "WFrequency", "SamplingFrequency", "Frequency", "Fs"):
        if key in g:
            try:
                v = g[key][()]
                return float(v)
            except Exception:
                pass
    return None


def _nex5_hdf5_summary(path: str) -> dict:
    import h5py

    out: dict = {"channels": []}
    with h5py.File(path, "r") as f:
        # Prefer standard 'Continuous' group; otherwise scan for groups with 1D 'Data'
        groups = []
        if "Continuous" in f and isinstance(f["Continuous"], h5py.Group):
            for name, g in f["Continuous"].items():
                if isinstance(g, h5py.Group) and "Data" in g and isinstance(g["Data"], h5py.Dataset):
                    groups.append((name, g))
        else:
            for name, g in f.items():
                if isinstance(g, h5py.Group) and "Data" in g and isinstance(g["Data"], h5py.Dataset):
                    groups.append((name, g))
        for i, (name, g) in enumerate(groups):
            fs = _h5_get_fs_from_group(g)
            units = g.attrs.get("Units", None)
            out["channels"].append({
                "index": i,
                "name": str(g.attrs.get("Name", name)),
                "path": f"{g.name}/Data",
                "sampling_rate": fs,
                "units": None if units is None else (units.decode("latin-1") if isinstance(units, bytes) else str(units)),
            })
    return out


def load_continuous_from_nex(
    filepath: str,
    *,
    channel_indices: Optional[List[int]] = None,
    duration_s: Optional[float] = None,
    fs_hint: Optional[float] = None,
) -> LFPTimeSeries:
    """Load continuous analog signals from a NeuroExplorer .nex/.nex5 file.

    Args:
        filepath: Absolute or relative path to a .nex/.nex5 file.
        channel_indices: Optional list of 0-based channel indices to keep.
            If None, all analog signals in the first segment are loaded.
        duration_s: Optional number of seconds to truncate from the start
            (useful to avoid loading the entire file into memory when only
            a short snippet is needed for QC/plots).
        fs_hint: Optional sampling rate (Hz) to use when the file header
            lacks a valid per-channel or stream sampling_rate.

    Returns:
        LFPTimeSeries: time series container with ``x`` shaped (T, C) and
        ``fs`` sampling rate. ``t`` is omitted for memory efficiency.

    Raises:
        FileNotFoundError: If the path does not exist.
        ImportError: If the required reader (neo) is not installed.
        RuntimeError: If no analog signals are found in the file.
        ValueError: If channel indices are invalid or inconsistent lengths.
    """
    p = Path(filepath)
    if not p.exists():
        raise FileNotFoundError(f"load_continuous_from_nex: path not found: {filepath}")

    # Import locally to keep dependency optional at package import time.
    from neo.rawio import NeuroExplorerRawIO  # type: ignore
    import neo.rawio.neuroexplorerrawio as nraw  # type: ignore
    import inspect  # stdlib
    import numpy as _np  # ensure numpy alias exists before reader

    # Helper: apply a tolerant decode patch to neo's read_as_dict at runtime
    # BEFORE parsing header, so we avoid the initial UnicodeDecodeError entirely.
    def _patch_nex5_decode() -> bool:
        src = inspect.getsource(nraw.read_as_dict)
        needs_patch = any(s in src for s in ("decode(\"utf8\")", "decode('utf8')", "decode(\"utf-8\")", "decode('utf-8')"))
        if not needs_patch:
            return True  # already tolerant or different implementation
        patched = (
            src.replace("decode(\"utf8\")", "decode('latin-1', errors='replace')")
            .replace("decode('utf8')", "decode('latin-1', errors='replace')")
            .replace("decode(\"utf-8\")", "decode('latin-1', errors='replace')")
            .replace("decode('utf-8')", "decode('latin-1', errors='replace')")
        )
        exec(patched, nraw.__dict__)
        return True

    _patch_nex5_decode()

    # Use RawIO to avoid constructing SpikeTrain objects, which can fail
    # on some files; we only need analog signals.
    reader = NeuroExplorerRawIO(filename=str(p))
    reader.parse_header()

    # Resolve channel table and sampling rates
    header = getattr(reader, "header", None)
    if header is None or "signal_channels" not in header:
        raise RuntimeError("No signal_channels found in NeuroExplorer file")
    sig = header["signal_channels"]
    if getattr(sig, "size", 0) == 0:
        raise RuntimeError("NeuroExplorer file has 0 analog signal channels")

    # sampling_rate can vary across streams; prefer per-channel when valid (>0),
    # otherwise use stream-level sampling_rate via stream_id mapping when available.
    names = sig.dtype.names
    per_ch_sr = None
    if "sampling_rate" in names:
        per_ch_sr = _np.asarray([float(v) for v in sig["sampling_rate"]])
    # Build stream sampling-rate map when possible
    stream_sr = {}
    if "signal_streams" in header and "stream_id" in names:
        ss = header["signal_streams"]
        for i in range(int(ss.size)):
            sid = str(ss["id"][i]) if "id" in ss.dtype.names else str(i)
            sr = float(ss["sampling_rate"][i]) if "sampling_rate" in ss.dtype.names else float("nan")
            stream_sr[sid] = sr
    # Choose SR for each channel
    sr_all_list = []
    for i in range(int(sig.size)):
        val = None
        if per_ch_sr is not None:
            val = float(per_ch_sr[i])
        if (val is None) or (not _np.isfinite(val)) or (val <= 0):
            if "stream_id" in names:
                sid = str(sig["stream_id"][i])
                if sid in stream_sr:
                    val = float(stream_sr[sid])
        if val is None or (not _np.isfinite(val)) or (val <= 0):
            val = float("nan")
        sr_all_list.append(val)
    sr_all = _np.asarray(sr_all_list)

    # Channel selection (absolute indices over signal_channels table)
    if channel_indices is None:
        chan_sel = list(range(int(sig.size)))
    else:
        chan_sel = list(channel_indices)
        if any((i < 0 or i >= int(sig.size)) for i in chan_sel):
            raise ValueError(f"channel_indices out of range; got {chan_sel} with C={int(sig.size)}")

    # Ensure all selected channels belong to the same stream (when stream_id present)
    stream_index = 0
    if "stream_id" in names:
        sel_sids = [str(sig["stream_id"][i]) for i in chan_sel]
        uniq_sid = sorted(set(sel_sids))
        if len(uniq_sid) != 1:
            raise ValueError(f"Selected channels span multiple streams: {uniq_sid}")
        sel_sid = uniq_sid[0]
        if "signal_streams" in header:
            ss = header["signal_streams"]
            ids = [str(ss["id"][i]) if "id" in ss.dtype.names else str(i) for i in range(int(ss.size))]
            try:
                stream_index = ids.index(sel_sid)
            except ValueError:
                stream_index = 0

    # Ensure a single sampling rate among selected channels
    sr_sel = sr_all[chan_sel]
    sr_unique = _np.unique(sr_sel)
    # Resolve sampling rate: prefer header value if valid; otherwise use hint
    if sr_unique.size != 1 or (not _np.isfinite(sr_unique[0])) or (sr_unique[0] <= 0):
        if fs_hint is None or fs_hint <= 0:
            raise ValueError(
                f"Selected channels have multiple or invalid sampling rates: {sr_unique.tolist()} "
                "and no fs_hint was provided. Supply --fs in the CLI to override."
            )
        fs = float(fs_hint)
    else:
        fs = float(sr_unique[0])

    # Determine read window
    i_start = 0
    if duration_s is not None:
        if duration_s <= 0:
            raise ValueError("duration_s must be positive when provided")
        i_stop: Optional[int] = int(duration_s * fs)
    else:
        i_stop = None

    # Read chunk (T, C)
    # Clamp i_stop to total available samples for the selected stream
    n_total = int(reader.get_signal_size(block_index=0, seg_index=0, stream_index=stream_index))
    if i_stop is None or i_stop > n_total:
        i_stop_eff = n_total
    else:
        i_stop_eff = i_stop

    X = reader.get_analogsignal_chunk(
        block_index=0,
        seg_index=0,
        i_start=i_start,
        i_stop=i_stop_eff,
        channel_indexes=chan_sel,
    )
    X = _np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] == 0:
        # HDF5 fallback for NEX5 files that RawIO fails to serve
        if _is_hdf5_file(str(p)):
            return _load_nex5_hdf5(str(p), channel_indices=channel_indices, duration_s=duration_s, fs_hint=fs)
        raise RuntimeError(
            "No analog samples read from file; verify --fs and --duration, and that the file contains continuous data."
        )

    # Channel names
    names = [str(n) for n in sig["name"][chan_sel]]

    return LFPTimeSeries(t=None, x=X, fs=fs, channel_ids=names, meta={"source": str(p)})


def _load_nex5_hdf5(
    filepath: str,
    *,
    channel_indices: Optional[List[int]],
    duration_s: Optional[float],
    fs_hint: Optional[float],
) -> LFPTimeSeries:
    import h5py

    p = Path(filepath)
    with h5py.File(str(p), "r") as f:
        groups = []
        if "Continuous" in f and isinstance(f["Continuous"], h5py.Group):
            for name, g in f["Continuous"].items():
                if isinstance(g, h5py.Group) and "Data" in g and isinstance(g["Data"], h5py.Dataset):
                    groups.append((name, g))
        else:
            for name, g in f.items():
                if isinstance(g, h5py.Group) and "Data" in g and isinstance(g["Data"], h5py.Dataset):
                    groups.append((name, g))
        if not groups:
            raise RuntimeError("HDF5: no continuous groups with a 1D 'Data' dataset found")

        # Channel selection mapping
        chan_all = list(range(len(groups)))
        if channel_indices is None:
            chan_sel = chan_all
        else:
            chan_sel = list(channel_indices)
            if any(i < 0 or i >= len(groups) for i in chan_sel):
                raise ValueError(f"HDF5: channel_indices out of range; got {chan_sel} with C={len(groups)}")

        # Resolve sampling rates; prefer per-group attribute; fallback to fs_hint
        fs_list: List[Optional[float]] = []
        for _, g in groups:
            fs_list.append(_h5_get_fs_from_group(g))
        fs_sel = [fs_list[i] for i in chan_sel]
        fs_vals = [float(v) for v in fs_sel if v is not None and not np.isnan(v) and v > 0]
        if fs_vals:
            fs = float(fs_vals[0])
            if any(abs(float(v) - fs) > 1e-6 for v in fs_vals):
                raise ValueError("HDF5: selected channels have differing sampling rates; pick channels from one rate or supply --fs")
        else:
            if fs_hint is None or fs_hint <= 0:
                raise ValueError("HDF5: sampling rate missing; supply --fs")
            fs = float(fs_hint)

        # Read data columns and stack; enforce same length across channels
        cols: List[np.ndarray] = []
        names: List[str] = []
        for i in chan_sel:
            name, g = groups[i]
            dset = g["Data"]
            a = np.asarray(dset, dtype=float).reshape(-1)
            cols.append(a)
            nm = g.attrs.get("Name", name)
            if isinstance(nm, bytes):
                nm = nm.decode("latin-1", errors="replace")
            names.append(str(nm))

        min_len = min(c.shape[0] for c in cols)
        if duration_s is not None:
            max_len = int(duration_s * fs)
            min_len = min(min_len, max_len)
        X = np.stack([c[:min_len] for c in cols], axis=1)
        if X.shape[0] == 0:
            raise RuntimeError("HDF5: zero samples after slicing; check --duration and fs")

    return LFPTimeSeries(t=None, x=X, fs=fs, channel_ids=names, meta={"source": str(p)})


def load_continuous_from_dir(
    dirpath: str,
    *,
    pattern: str = "*.nex5",
    duration_s: Optional[float] = 10.0,
    fs_hint: Optional[float] = None,
) -> LFPTimeSeries:
    """Aggregate continuous channels from multiple NEX/NEX5 files in a directory.

    Each file is loaded via ``load_continuous_from_nex`` and treated as one or
    more channels. Channels across files are concatenated column-wise, and the
    time axis is truncated to the shortest among loaded files (or to
    ``duration_s`` if provided).

    Args:
        dirpath: Directory containing .nex/.nex5 files.
        pattern: Glob to select files (default: '*.nex5').
        duration_s: Seconds to load from the start (default: 10s). When None,
            loads the full available duration and truncates to the shortest
            across files (may be memory-heavy).
        fs_hint: Sampling rate (Hz) to use if files lack a valid fs.

    Returns:
        LFPTimeSeries with shape (T, C_total) across all matched files.

    Raises:
        FileNotFoundError: If the directory is missing or no files match.
        ValueError: If sampling rates differ across files without an override.
        RuntimeError: If no samples can be read from any file.
    """
    d = Path(dirpath)
    if not d.exists() or not d.is_dir():
        raise FileNotFoundError(f"load_continuous_from_dir: directory not found: {dirpath}")

    files = sorted(d.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files match pattern '{pattern}' in {dirpath}")

    lfps: List[LFPTimeSeries] = []
    for f in files:
        lfp = load_continuous_from_nex(str(f), duration_s=duration_s, fs_hint=fs_hint)
        lfps.append(lfp)

    # Validate sampling rates
    fs_vals = [float(l.fs or 0.0) for l in lfps]
    fs_uniq = sorted(set(fs_vals))
    if len(fs_uniq) != 1 or fs_uniq[0] <= 0:
        raise ValueError(f"Files have multiple or invalid sampling rates: {fs_uniq}")
    fs = fs_uniq[0]

    # Truncate all to the minimum length
    min_len = min(l.x.shape[0] for l in lfps)
    if min_len <= 0:
        raise RuntimeError("No samples read from any file in directory")

    cols: List[np.ndarray] = []
    names: List[str] = []
    for l in lfps:
        X = l.x[:min_len, :]
        cols.append(X)
        if l.channel_ids:
            names.extend(list(l.channel_ids))
        else:
            # Fallback names
            names.extend([f"ch{k}" for k in range(X.shape[1])])

    Xcat = np.concatenate(cols, axis=1)
    return LFPTimeSeries(t=None, x=Xcat, fs=fs, channel_ids=names, meta={"source": str(d)})
