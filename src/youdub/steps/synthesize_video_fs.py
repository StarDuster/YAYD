from __future__ import annotations

import os


def _mtime(path: str) -> float | None:
    try:
        return float(os.path.getmtime(path))
    except Exception:
        return None


def _is_stale(target: str, deps: list[str]) -> bool:
    """
    Return True when `target` is missing/older than any dependency in `deps`.

    Dependencies missing/unreadable are ignored (best-effort).
    """
    t = _mtime(target)
    if t is None:
        return True
    for p in deps:
        mt = _mtime(p)
        if mt is None:
            continue
        if mt > t:
            return True
    return False

