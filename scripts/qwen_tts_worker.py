#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np


READY_LINE = "__READY__"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen3-TTS worker process (stdin/stdout JSONL protocol).")
    parser.add_argument("--model-path", required=True, help="Local model directory for Qwen3-TTS Base weights.")
    parser.add_argument("--stub", action="store_true", help="Run in stub mode (no qwen-tts import, for tests).")
    return parser.parse_args()


def _write_wav(path: str, wav: np.ndarray, sr: int) -> None:
    import soundfile as sf

    Path(os.path.dirname(path) or ".").mkdir(parents=True, exist_ok=True)
    sf.write(path, wav.astype(np.float32), int(sr))


def main() -> int:
    args = _parse_args()
    model_path = args.model_path

    # In stub mode we don't import qwen_tts at all; useful for CI/tests.
    tts = None
    prompt_cache: dict[str, object] = {}

    if not args.stub:
        import torch
        from qwen_tts import Qwen3TTSModel

        device_map = "cuda:0" if torch.cuda.is_available() else "cpu"
        if device_map.startswith("cuda") and torch.cuda.is_available():
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = torch.float32

        tts = Qwen3TTSModel.from_pretrained(
            model_path,
            device_map=device_map,
            dtype=dtype,
        )

    # Signal parent that we're ready.
    print(READY_LINE, flush=True)

    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            req = json.loads(raw)
        except Exception:
            print(json.dumps({"ok": False, "error": "invalid json"}), flush=True)
            continue

        cmd = req.get("cmd")
        if cmd == "shutdown":
            break
        if cmd != "synthesize":
            print(json.dumps({"ok": False, "error": f"unknown cmd: {cmd}"}), flush=True)
            continue

        try:
            text = str(req.get("text", ""))
            language = str(req.get("language", "Auto") or "Auto")
            speaker_wav = str(req.get("speaker_wav", ""))
            output_path = str(req.get("output_path", ""))
            if not speaker_wav or not output_path:
                raise ValueError("missing speaker_wav/output_path")

            if args.stub:
                sr = 24000
                wav = np.zeros(int(sr * 0.2), dtype=np.float32)
            else:
                assert tts is not None
                key = os.path.abspath(speaker_wav)
                prompt = prompt_cache.get(key)
                if prompt is None:
                    prompt = tts.create_voice_clone_prompt(
                        ref_audio=speaker_wav,
                        ref_text="",
                        x_vector_only_mode=True,
                    )
                    prompt_cache[key] = prompt

                wavs, sr = tts.generate_voice_clone(
                    text=text,
                    language=language,
                    voice_clone_prompt=prompt,
                )
                if not wavs:
                    raise RuntimeError("empty wav list")
                wav = np.asarray(wavs[0], dtype=np.float32)

            _write_wav(output_path, wav, int(sr))
            print(json.dumps({"ok": True, "output_path": output_path, "sr": int(sr), "n_samples": int(wav.shape[0])}), flush=True)
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            tb = traceback.format_exc(limit=8)
            print(json.dumps({"ok": False, "error": err, "trace": tb}, ensure_ascii=False), flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

