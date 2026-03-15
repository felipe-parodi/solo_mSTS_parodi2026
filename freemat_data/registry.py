from __future__ import annotations

"""Session registry that reads configs/sessions.yaml and provides normalized views."""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional
from pathlib import Path

import yaml
import re

from .types import SessionRef
from .pathing import normalize


@dataclass
class _Config:
    path: Path
    obj: Dict[str, Any]


def _sanitize_windows_paths(yaml_text: str) -> str:
    """Replace double-quoted Windows paths with single-quoted, forward-slashed.

    Example: key: "A:\\EnclosureProjects\\..." -> key: 'A:/EnclosureProjects/...'
    Only touches values that look like Windows absolute paths inside double quotes.
    """
    pattern = re.compile(r'(:\s*)"([A-Za-z]:\\[^"\n]+)"')

    def repl(m: re.Match) -> str:
        prefix = m.group(1)
        path = m.group(2)
        # Avoid backslashes inside f-string expressions (not allowed). Do sanitization first.
        sanitized = path.replace("\\", "/")
        return prefix + "'" + sanitized + "'"

    return pattern.sub(repl, yaml_text)


def _safe_load_yaml_file(path: Path) -> Dict[str, Any]:
    txt = path.read_text(encoding='utf-8')
    try:
        return yaml.safe_load(txt)
    except yaml.YAMLError:
        sanitized = _sanitize_windows_paths(txt)
        if sanitized != txt:
            try:
                return yaml.safe_load(sanitized)
            except yaml.YAMLError as e2:
                raise yaml.YAMLError(
                    f"Failed to parse YAML at {path}. If using Windows paths, prefer single quotes or forward slashes. Error: {e2}"
                )
        raise


class Registry:
    """Load and query session definitions across projects.

    This wraps the existing YAML format used by data.SessionManager and exposes
    a minimal typed API for consumers in freemat, freemat/single, and torch_brain.
    """

    def __init__(self, config_path: str = "configs/sessions.yaml") -> None:
        p = Path(config_path)
        if not p.exists():
            raise FileNotFoundError(f"Config not found: {p}")
        obj = _safe_load_yaml_file(p)
        self._cfg = _Config(path=p, obj=obj)

        # Normalize a quick sessions dict
        self._sessions: Dict[str, SessionRef] = {}
        for sid, sconf in (obj.get("sessions") or {}).items():
            data_files = dict(sconf.get("data_files") or {})
            # Normalize known keys – keep others as-is
            norm_files: Dict[str, Optional[str]] = {}
            for k, v in data_files.items():
                nv = normalize(v)
                norm_files[k] = str(nv) if nv is not None else None
            # Merge select top-level fields into metadata for convenience
            meta = dict(sconf.get("metadata") or {})
            if "subject" in sconf and "subject" not in meta:
                meta["subject"] = sconf.get("subject")
            if "logger" in sconf and "logger" not in meta:
                meta["logger"] = sconf.get("logger")
            if "reference_camera" in sconf and "reference_camera" not in meta:
                meta["reference_camera"] = sconf.get("reference_camera")
            if "units" in sconf and "units" not in meta:
                meta["units"] = sconf.get("units")

            self._sessions[sid] = SessionRef(
                session_id=sid,
                session_type=str(sconf.get("type", "social")),
                monkeys=list(sconf.get("monkeys") or []),
                data_files=norm_files,
                metadata=meta,
            )

    def get(self, session_id: str) -> Optional[SessionRef]:
        return self._sessions.get(session_id)

    def list_sessions(
        self,
        *,
        session_type: str = "all",
        has: Optional[Iterable[str]] = None,
    ) -> List[SessionRef]:
        out: List[SessionRef] = []
        for s in self._sessions.values():
            if session_type != "all" and s.session_type != session_type:
                continue
            if has:
                ok = True
                for key in has:
                    # accept either exact keys or canonical variants
                    if key == "pose3d" and not (s.data_files.get("pose_3d") or s.data_files.get("pose3d_npz")):
                        ok = False
                        break
                    if key in ("spikes", "neural") and not any(k.endswith("_neural") for k in s.data_files.keys()):
                        ok = False
                        break
                if not ok:
                    continue
            out.append(s)
        return out


def load_session_yaml(config_path: str) -> SessionRef:
    """Load a single-session YAML into a SessionRef.

    Accepts either:
      - a per-session file with keys: session_id, type, monkeys, data_files, metadata
      - a multi-session file with exactly one entry under 'sessions'
    """
    p = Path(config_path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    obj = _safe_load_yaml_file(p)

    # Multi-session with a single entry
    if isinstance(obj, dict) and isinstance(obj.get("sessions"), dict):
        sessions = obj["sessions"]
        if len(sessions) != 1:
            raise ValueError("Expected exactly one session in 'sessions' for per-session YAML")
        sid, sconf = next(iter(sessions.items()))
    else:
        # Per-session at top level
        meta_block = obj.get("metadata") or {}
        sid = str(obj.get("session_id") or obj.get("id") or meta_block.get("session_id") or "session")
        sconf = obj

    data_files = dict(sconf.get("data_files") or {})
    norm_files: Dict[str, Optional[str]] = {}
    for k, v in data_files.items():
        nv = normalize(v)
        norm_files[k] = str(nv) if nv is not None else None

    # Merge top-level fields into metadata for convenience
    meta = dict(sconf.get("metadata") or {})
    if "subject" in sconf and "subject" not in meta:
        meta["subject"] = sconf.get("subject")
    if "logger" in sconf and "logger" not in meta:
        meta["logger"] = sconf.get("logger")
    if "reference_camera" in sconf and "reference_camera" not in meta:
        meta["reference_camera"] = sconf.get("reference_camera")
    if "units" in sconf and "units" not in meta:
        meta["units"] = sconf.get("units")

    session_type = str(sconf.get("type") or meta.get("type") or "asocial")
    monkeys = list(sconf.get("monkeys") or meta.get("monkeys") or [])

    return SessionRef(
        session_id=sid,
        session_type=session_type,
        monkeys=monkeys,
        data_files=norm_files,
        metadata=meta,
    )
