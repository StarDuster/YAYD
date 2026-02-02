"""Syllable nuclei detector (phenom stats) tests."""

from __future__ import annotations

import numpy as np


def _add_tone_burst(
    y: np.ndarray,
    sr: int,
    *,
    start_s: float,
    dur_s: float = 0.12,
    f0: float = 200.0,
    amp: float = 0.8,
) -> None:
    n = int(round(float(dur_s) * float(sr)))
    if n <= 4:
        return
    i0 = int(round(float(start_s) * float(sr)))
    i1 = min(int(y.shape[0]), i0 + n)
    if i0 < 0 or i0 >= i1:
        return

    t = (np.arange(i1 - i0, dtype=np.float32) / float(sr)).astype(np.float32, copy=False)
    tone = np.sin(2.0 * np.pi * float(f0) * t).astype(np.float32, copy=False)
    win = np.hanning(i1 - i0).astype(np.float32, copy=False)
    burst = (tone * win * float(amp)).astype(np.float32, copy=False)
    y[i0:i1] += burst


def test_estimate_syllable_nuclei_count_detects_bursts():
    import youdub.steps.synthesize_video as sv

    sr = 16000
    dur = 2.0
    y = np.zeros(int(round(float(sr) * float(dur))), dtype=np.float32)

    times = [0.25, 0.65, 1.05, 1.45]
    for t0 in times:
        _add_tone_burst(y, sr, start_s=float(t0))

    # Use voiced_filter=False to keep this test deterministic and fast.
    n = sv._estimate_syllable_nuclei_count(y, sr, voiced_filter=False)
    assert n == len(times)


def test_estimate_syllable_nuclei_count_silence_is_zero():
    import youdub.steps.synthesize_video as sv

    sr = 16000
    y = np.zeros(sr, dtype=np.float32)
    n = sv._estimate_syllable_nuclei_count(y, sr, voiced_filter=False)
    assert n == 0


def test_estimate_syllable_nuclei_count_voiced_filter_does_not_crash():
    import youdub.steps.synthesize_video as sv

    sr = 16000
    y = np.zeros(int(sr * 1.2), dtype=np.float32)
    _add_tone_burst(y, sr, start_s=0.25)
    _add_tone_burst(y, sr, start_s=0.75)

    n = sv._estimate_syllable_nuclei_count(y, sr, voiced_filter=True)
    assert isinstance(n, int)
    assert n >= 0

