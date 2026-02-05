from __future__ import annotations

import sys as _sys

from .tts import qwen_worker as _impl

_sys.modules[__name__] = _impl

