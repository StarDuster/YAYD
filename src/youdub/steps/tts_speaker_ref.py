from __future__ import annotations

import sys as _sys

from .tts import speaker_ref as _impl

_sys.modules[__name__] = _impl

