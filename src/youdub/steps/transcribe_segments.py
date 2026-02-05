from __future__ import annotations

import sys as _sys

from .transcribe import segments as _impl

_sys.modules[__name__] = _impl

