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

        if torch.cuda.is_available():
            device_map = "cuda:0"
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            tts = Qwen3TTSModel.from_pretrained(
                model_path,
                device_map=device_map,
                dtype=dtype,
            )
        else:
            # CPU 上直接 float32 加载 1.7B 权重很容易 OOM 被系统杀掉（尤其在 WSL2 默认内存限制下）。
            # 这里优先尝试更省内存的 dtype；如果仍失败，直接抛错让父进程拿到可读的 traceback。
            device_map = "cpu"
            last_exc: Exception | None = None
            for dtype in (torch.bfloat16, torch.float16, torch.float32):
                try:
                    tts = Qwen3TTSModel.from_pretrained(
                        model_path,
                        device_map=device_map,
                        dtype=dtype,
                    )
                    last_exc = None
                    break
                except Exception as exc:  # pylint: disable=broad-except
                    last_exc = exc
                    tts = None
            if tts is None:
                raise RuntimeError(
                    "无法在 CPU 上加载 Qwen3-TTS 模型（已尝试 bf16/fp16/fp32）。"
                    "建议启用 CUDA/GPU，或换更小/量化的模型权重。"
                ) from last_exc

    # Signal parent that we're ready.
    print(READY_LINE, flush=True)

    def _error_result(exc: Exception) -> dict:
        err = f"{type(exc).__name__}: {exc}"
        tb = traceback.format_exc(limit=8)
        return {"ok": False, "error": err, "trace": tb}

    def _get_prompt(speaker_wav: str):
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
        return prompt

    try:
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
            if cmd not in {"synthesize", "synthesize_batch"}:
                print(json.dumps({"ok": False, "error": f"unknown cmd: {cmd}"}), flush=True)
                continue

            try:
                if cmd == "synthesize":
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
                        prompt = _get_prompt(speaker_wav)
                        wavs, sr = tts.generate_voice_clone(  # type: ignore[union-attr]
                            text=text,
                            language=language,
                            voice_clone_prompt=prompt,
                        )
                        if not wavs:
                            raise RuntimeError("empty wav list")
                        wav = np.asarray(wavs[0], dtype=np.float32)

                    _write_wav(output_path, wav, int(sr))
                    print(
                        json.dumps(
                            {"ok": True, "output_path": output_path, "sr": int(sr), "n_samples": int(wav.shape[0])}
                        ),
                        flush=True,
                    )
                    continue

                # cmd == "synthesize_batch"
                items = req.get("items", None)
                if not isinstance(items, list) or not items:
                    raise ValueError("missing items")

                # Collect and validate
                parsed: list[tuple[int, str, str, str, str]] = []
                for idx, it in enumerate(items):
                    if not isinstance(it, dict):
                        raise ValueError(f"items[{idx}] is not an object")
                    text = str(it.get("text", ""))
                    language = str(it.get("language", "Auto") or "Auto")
                    speaker_wav = str(it.get("speaker_wav", ""))
                    output_path = str(it.get("output_path", ""))
                    if not speaker_wav or not output_path:
                        raise ValueError(f"items[{idx}] missing speaker_wav/output_path")
                    parsed.append((idx, text, language, speaker_wav, output_path))

                results: list[dict] = [{"ok": False, "error": "not processed"} for _ in parsed]

                if args.stub:
                    sr = 24000
                    for idx, _text, _lang, _spk, out in parsed:
                        wav = np.zeros(int(sr * 0.2), dtype=np.float32)
                        _write_wav(out, wav, int(sr))
                        results[idx] = {"ok": True, "output_path": out, "sr": int(sr), "n_samples": int(wav.shape[0])}
                    print(json.dumps({"ok": True, "results": results}, ensure_ascii=False), flush=True)
                    continue

                assert tts is not None
                # Group by speaker_wav to maximize prompt reuse.
                by_spk: dict[str, list[tuple[int, str, str, str]]] = {}
                for idx, text, language, speaker_wav, output_path in parsed:
                    key = os.path.abspath(speaker_wav)
                    by_spk.setdefault(key, []).append((idx, text, language, output_path))

                for key, group in by_spk.items():
                    speaker_wav = key
                    try:
                        prompt = _get_prompt(speaker_wav)
                        texts = [t for (_idx, t, _lang, _out) in group]
                        langs = [lang for (_idx, _t, lang, _out) in group]
                        wavs, sr = tts.generate_voice_clone(  # type: ignore[union-attr]
                            text=texts,
                            language=langs,
                            voice_clone_prompt=prompt,
                        )
                        if not wavs:
                            raise RuntimeError("empty wav list")
                        if len(wavs) != len(group):
                            raise RuntimeError(f"batch wav count mismatch: got {len(wavs)}, expected {len(group)}")

                        for wav, (idx, _text, _lang, out) in zip(wavs, group):
                            wav_np = np.asarray(wav, dtype=np.float32)
                            _write_wav(out, wav_np, int(sr))
                            results[idx] = {
                                "ok": True,
                                "output_path": out,
                                "sr": int(sr),
                                "n_samples": int(wav_np.shape[0]),
                            }
                    except Exception as exc:
                        err_res = _error_result(exc)
                        for idx, _text, _lang, _out in group:
                            results[idx] = {
                                "ok": False,
                                "error": err_res.get("error", "unknown"),
                                "trace": err_res.get("trace"),
                            }

                print(json.dumps({"ok": True, "results": results}, ensure_ascii=False), flush=True)
            except Exception as exc:
                print(json.dumps(_error_result(exc), ensure_ascii=False), flush=True)

        return 0
    except KeyboardInterrupt:
        # Quiet exit on Ctrl+C (parent will handle cancellation).
        return 130


if __name__ == "__main__":
    raise SystemExit(main())

