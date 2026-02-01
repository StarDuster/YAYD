"""Regression tests for PyTorch weights_only compatibility shim."""

from __future__ import annotations

import pytest


class _PickleSentinel:
    def __init__(self, x: int):
        self.x = int(x)


def test_torch_load_weights_only_compat_allows_legacy_pickles(tmp_path):
    import torch

    from youdub.utils import torch_load_weights_only_compat

    p = tmp_path / "obj.pt"
    torch.save(_PickleSentinel(123), p)

    # PyTorch 2.6+ uses weights_only=True by default and restricts unpickling.
    # Verify that forcing weights_only=True fails for an object with a custom global.
    try:
        torch.load(p, weights_only=True)
    except TypeError:
        pytest.skip("当前 torch 版本不支持 weights_only 参数")
    except Exception:
        pass
    else:
        pytest.skip("weights_only=True 意外成功，当前环境不需要该兼容")

    with torch_load_weights_only_compat():
        loaded = torch.load(p, weights_only=True)

    assert isinstance(loaded, _PickleSentinel)
    assert loaded.x == 123

