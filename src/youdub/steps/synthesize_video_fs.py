from __future__ import annotations

import sys as _sys

from .video import fs as _impl

_sys.modules[__name__] = _impl

