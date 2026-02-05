from __future__ import annotations

import sys as _sys

from .transcribe import asr_run as _impl

_sys.modules[__name__] = _impl

