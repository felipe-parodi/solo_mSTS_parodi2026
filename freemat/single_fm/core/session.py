"""Session utilities for Single-FM.

Provide a single entry to resolve a session from either a per-session
YAML or a registry YAML + session_id.
"""

from __future__ import annotations

from typing import Optional

from freemat_data import Registry, SessionRef, load_session_yaml


def resolve_session(
    *, session_yaml: Optional[str] = None, sessions_yaml: Optional[str] = None, session_id: Optional[str] = None
) -> SessionRef:
    """Resolve SessionRef from YAML or registry.

    Args:
        session_yaml: Path to per-session YAML
        sessions_yaml: Path to multi-session registry YAML
        session_id: Session ID within the registry

    Returns:
        SessionRef object

    Raises:
        ValueError: If arguments are insufficient or session is missing
    """

    if session_yaml:
        return load_session_yaml(session_yaml)
    if not sessions_yaml or not session_id:
        raise ValueError("resolve_session: provide session_yaml or (sessions_yaml + session_id)")
    reg = Registry(sessions_yaml)
    sess = reg.get(session_id)
    if sess is None:
        raise ValueError(f"resolve_session: unknown session_id in registry: {session_id}")
    return sess


__all__ = ["resolve_session"]

