import sys
from pathlib import Path


def test_qwen_worker_protocol_tolerates_stdout_noise(tmp_path, monkeypatch):
    """
    Regression test:
    - Worker may print non-protocol lines to stdout before __READY__ (or between request/response).
    - Parent must ignore noise and still complete handshake + JSON response parsing.
    """
    import youdub.core.steps.synthesize_speech as ss
    from youdub.config import Settings
    from youdub.models import ModelManager

    # Ensure we don't accidentally pass --stub from environment.
    monkeypatch.delenv("YOUDUB_QWEN_WORKER_STUB", raising=False)

    # Create a tiny "noisy worker" compatible with the same argv protocol.
    worker_path = tmp_path / "noisy_qwen_worker.py"
    worker_path.write_text(
        """#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import wave

READY_LINE = "__READY__"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--stub", action="store_true")
    return p.parse_args()


def _write_wav(path: str, sr: int = 24000, seconds: float = 0.2) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    n = int(sr * seconds)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(b"\\x00\\x00" * n)


def main() -> int:
    _ = _parse_args()

    # Noise before READY (stdout + stderr)
    print("noise-before-ready", flush=True)
    print("stderr-noise-before-ready", file=sys.stderr, flush=True)
    print(READY_LINE, flush=True)

    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        req = json.loads(raw)
        cmd = req.get("cmd")
        if cmd == "shutdown":
            break
        if cmd != "synthesize":
            print(json.dumps({"ok": False, "error": f"unknown cmd: {cmd}"}), flush=True)
            continue

        # Noise between request and response (stdout)
        print("noise-between-request-and-response", flush=True)

        out = str(req.get("output_path", ""))
        _write_wav(out)
        print(json.dumps({"ok": True, "output_path": out}), flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
""",
        encoding="utf-8",
    )

    # Use our worker script for this test only.
    monkeypatch.setattr(ss, "_get_qwen_worker_script_path", lambda: worker_path)

    # Minimal settings: python exists; model dir exists (worker ignores it but arg is required).
    model_dir = tmp_path / "dummy_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "dummy.txt").write_text("ok", encoding="utf-8")

    test_settings = Settings(qwen_tts_model_path=model_dir, qwen_tts_python_path=Path(sys.executable))
    monkeypatch.setattr(ss, "_DEFAULT_SETTINGS", test_settings)
    monkeypatch.setattr(ss, "_DEFAULT_MODEL_MANAGER", ModelManager(test_settings))

    w = ss._QwenTtsWorker.from_settings(test_settings)  # noqa: SLF001
    out = tmp_path / "out.wav"
    w.synthesize("hello", speaker_wav=str(tmp_path / "spk.wav"), output_path=str(out), language="Auto")
    w.close()

    assert out.exists()
    assert out.stat().st_size >= 44

