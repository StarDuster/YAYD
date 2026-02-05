from __future__ import annotations

import ctypes
import importlib.util
import sys
from pathlib import Path

from loguru import logger


_CUDNN_PRELOADED = False


def preload_cudnn_for_onnxruntime_gpu() -> None:
    """Ensure onnxruntime-gpu can find cuDNN shipped as pip wheels.

    In some environments (notably WSL / minimal containers), CUDA libraries are available
    system-wide, but cuDNN is installed via `nvidia-cudnn-cu12` inside the venv only.
    `onnxruntime-gpu` loads `libonnxruntime_providers_cuda.so`, which depends on `libcudnn.so.9`.
    The dynamic loader won't search the venv site-packages directory by default, leading to
    a hard crash/abort in downstream libraries.

    We preload `libcudnn.so.9` via an absolute path and RTLD_GLOBAL so that subsequent loads
    can resolve the dependency by SONAME.
    """
    global _CUDNN_PRELOADED  # noqa: PLW0603

    if _CUDNN_PRELOADED:
        return
    _CUDNN_PRELOADED = True

    if sys.platform != "linux":
        return

    try:
        ort_spec = importlib.util.find_spec("onnxruntime")
        if not ort_spec or not ort_spec.submodule_search_locations:
            return
        ort_root = Path(list(ort_spec.submodule_search_locations)[0])
        if not (ort_root / "capi" / "libonnxruntime_providers_cuda.so").exists():
            # CPU-only build; nothing to do.
            return

        cudnn_spec = importlib.util.find_spec("nvidia.cudnn")
        if not cudnn_spec or not cudnn_spec.submodule_search_locations:
            return
        cudnn_root = Path(list(cudnn_spec.submodule_search_locations)[0])
        cudnn_lib = cudnn_root / "lib" / "libcudnn.so.9"
        if not cudnn_lib.exists():
            return

        ctypes.CDLL(str(cudnn_lib), mode=ctypes.RTLD_GLOBAL)
        logger.info(f"已预加载cuDNN用于onnxruntime-gpu: {cudnn_lib}")
    except Exception as exc:  # pylint: disable=broad-except
        # Never hard-fail here; downstream may still work if the system has cuDNN installed.
        logger.warning(f"预加载cuDNN失败 onnxruntime-gpu: {exc}")


_TORCHAUDIO_BACKEND_COMPAT_PATCHED = False


def patch_torchaudio_backend_compat() -> None:
    """
    torchaudio 2.10+ removed legacy backend APIs:
      - torchaudio.list_audio_backends
      - torchaudio.get_audio_backend
      - torchaudio.set_audio_backend

    Some downstream libs (notably speechbrain imported by pyannote.audio pipelines) still call them.
    We inject a tiny compatibility shim so those calls won't crash.

    NOTE: This does NOT actually switch torchaudio I/O backends (2.10+ doesn't expose that
    mechanism publicly). It only prevents AttributeError and records the requested backend.
    """
    global _TORCHAUDIO_BACKEND_COMPAT_PATCHED  # noqa: PLW0603

    if _TORCHAUDIO_BACKEND_COMPAT_PATCHED:
        return
    _TORCHAUDIO_BACKEND_COMPAT_PATCHED = True

    try:
        import torchaudio  # type: ignore
    except Exception:
        return

    if (
        hasattr(torchaudio, "list_audio_backends")
        and hasattr(torchaudio, "get_audio_backend")
        and hasattr(torchaudio, "set_audio_backend")
    ):
        return

    backends: list[str] = []
    try:
        import soundfile  # noqa: F401

        backends.append("soundfile")
    except Exception:
        pass

    # Attach best-effort state onto torchaudio module.
    try:
        setattr(torchaudio, "_youdub_audio_backends", backends)
        setattr(torchaudio, "_youdub_audio_backend", backends[0] if backends else None)
    except Exception:
        # If we cannot attach state, still patch functions (they'll return defaults).
        pass

    def _list_audio_backends() -> list[str]:
        try:
            v = getattr(torchaudio, "_youdub_audio_backends", None)
            return list(v) if isinstance(v, list) else []
        except Exception:
            return []

    def _get_audio_backend() -> str | None:
        try:
            v = getattr(torchaudio, "_youdub_audio_backend", None)
            return str(v) if v is not None else None
        except Exception:
            return None

    def _set_audio_backend(name: object) -> None:
        try:
            s = str(name).strip() if name is not None else ""
        except Exception:
            s = ""
        try:
            setattr(torchaudio, "_youdub_audio_backend", s or None)
        except Exception:
            return

    # Inject
    try:
        if not hasattr(torchaudio, "list_audio_backends"):
            setattr(torchaudio, "list_audio_backends", _list_audio_backends)
        if not hasattr(torchaudio, "get_audio_backend"):
            setattr(torchaudio, "get_audio_backend", _get_audio_backend)
        if not hasattr(torchaudio, "set_audio_backend"):
            setattr(torchaudio, "set_audio_backend", _set_audio_backend)
        logger.info("已为 torchaudio 注入 backend 兼容 API（list/get/set_audio_backend），用于 pyannote 兼容。")
    except Exception:
        return

