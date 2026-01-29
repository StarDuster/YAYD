from __future__ import annotations

import signal
import threading
import time
from contextlib import contextmanager
from typing import Iterator

from loguru import logger

_CANCEL_EVENT = threading.Event()
_CANCEL_REASON: str | None = None
_HANDLERS_INSTALLED = False
_SIGINT_COUNT = 0


class CancelledByUser(BaseException):
    """Raised when user requests cancellation (Ctrl+C / SIGTERM)."""

    def __init__(self, reason: str | None = None):
        super().__init__(reason or "cancelled")
        self.reason = reason or "cancelled"


def request_cancel(reason: str | None = None) -> None:
    """Mark the current process as cancelled (idempotent)."""
    global _CANCEL_REASON  # noqa: PLW0603

    if _CANCEL_EVENT.is_set():
        return
    _CANCEL_REASON = (reason or "cancelled").strip() or "cancelled"
    _CANCEL_EVENT.set()
    try:
        logger.warning(f"已请求取消: {_CANCEL_REASON}")
    except Exception:
        # Logging must never block cancellation.
        pass


def cancel_requested() -> bool:
    return _CANCEL_EVENT.is_set()


def cancel_reason() -> str | None:
    return _CANCEL_REASON


def reset_cancel() -> None:
    """Reset cancellation state for a new task."""
    global _CANCEL_REASON, _SIGINT_COUNT  # noqa: PLW0603
    _CANCEL_EVENT.clear()
    _CANCEL_REASON = None
    _SIGINT_COUNT = 0


def check_cancelled(reason: str | None = None) -> None:
    """Raise CancelledByUser if cancellation has been requested."""
    if _CANCEL_EVENT.is_set():
        raise CancelledByUser(reason or _CANCEL_REASON)


def sleep_with_cancel(seconds: float, step: float = 0.2) -> None:
    """Sleep but remain responsive to cancellation."""
    end = time.monotonic() + max(0.0, float(seconds))
    while True:
        check_cancelled()
        now = time.monotonic()
        if now >= end:
            return
        time.sleep(min(float(step), end - now))


@contextmanager
def ignore_signals_during_shutdown() -> Iterator[None]:
    """Ignore SIGINT/SIGTERM while shutting down (prevents double Ctrl+C crashes)."""
    old: dict[int, object] = {}
    for sig in (getattr(signal, "SIGINT", None), getattr(signal, "SIGTERM", None)):
        if sig is None:
            continue
        try:
            old[int(sig)] = signal.getsignal(sig)  # type: ignore[arg-type]
            signal.signal(sig, signal.SIG_IGN)  # type: ignore[arg-type]
        except Exception:
            pass
    try:
        yield
    finally:
        for sig, handler in old.items():
            try:
                signal.signal(sig, handler)  # type: ignore[arg-type]
            except Exception:
                pass


def install_signal_handlers() -> None:
    """Install signal handlers to request cancellation.

    We keep default SIGINT behavior (KeyboardInterrupt) by chaining to the previous handler.
    """
    global _HANDLERS_INSTALLED  # noqa: PLW0603
    if _HANDLERS_INSTALLED:
        return
    _HANDLERS_INSTALLED = True

    try:
        sigint = signal.SIGINT
        sigterm = signal.SIGTERM
    except Exception:
        return

    prev_int = signal.getsignal(sigint)
    prev_term = signal.getsignal(sigterm)

    def _handler(signum: int, frame) -> None:  # noqa: ARG001
        global _SIGINT_COUNT  # noqa: PLW0603
        name = "SIGINT" if signum == sigint else "SIGTERM" if signum == sigterm else str(signum)

        if signum == sigint:
            _SIGINT_COUNT += 1
            if _SIGINT_COUNT >= 2:
                # Second Ctrl+C: force exit immediately (blocking I/O won't respond to signals)
                import os
                import sys

                try:
                    logger.warning("第二次 Ctrl+C，强制退出")
                except Exception:
                    pass
                try:
                    sys.stderr.write("\n强制退出\n")
                    sys.stderr.flush()
                except Exception:
                    pass
                os._exit(130)

        request_cancel(name)

        prev = prev_int if signum == sigint else prev_term
        if callable(prev):
            prev(signum, frame)
            return
        if prev == signal.SIG_DFL:
            if signum == sigint:
                raise KeyboardInterrupt
            raise SystemExit(128 + int(signum))
        return

    try:
        signal.signal(sigint, _handler)
    except Exception:
        pass
    try:
        signal.signal(sigterm, _handler)
    except Exception:
        pass

