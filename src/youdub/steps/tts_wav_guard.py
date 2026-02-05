from __future__ import annotations

import sys as _sys

from .tts import wav_guard as _impl

_sys.modules[__name__] = _impl

