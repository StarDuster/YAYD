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
DEFAULT_MAX_NEW_TOKENS = 2048


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
    xvec_prompt_cache: dict[str, object] = {}
    # Cap generation length to avoid pathological endless outputs.
    try:
        max_new_tokens = int(os.getenv("YOUDUB_QWEN_TTS_MAX_NEW_TOKENS", str(DEFAULT_MAX_NEW_TOKENS)) or DEFAULT_MAX_NEW_TOKENS)
    except Exception:
        max_new_tokens = int(DEFAULT_MAX_NEW_TOKENS)

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

    def _get_prompt_xvec_only(speaker_wav: str):
        assert tts is not None
        key = os.path.abspath(speaker_wav)
        prompt = xvec_prompt_cache.get(key)
        if prompt is None:
            prompt = tts.create_voice_clone_prompt(
                ref_audio=speaker_wav,
                ref_text="",
                x_vector_only_mode=True,
            )
            xvec_prompt_cache[key] = prompt
        return prompt

    try:
        for raw in sys.stdin:
            raw = raw.strip()
            if not raw:
                continue
            try:
                req = json.loads(raw)
            except Exception:
                print(json.dumps({"ok": False, "error": "JSON无效"}), flush=True)
                continue

            cmd = req.get("cmd")
            if cmd == "shutdown":
                break
            if cmd not in {"synthesize", "synthesize_batch"}:
                print(json.dumps({"ok": False, "error": f"未知命令: {cmd}"}), flush=True)
                continue

            try:
                if cmd == "synthesize":
                    text = str(req.get("text", ""))
                    language = str(req.get("language", "Auto") or "Auto")
                    speaker_wav = str(req.get("speaker_wav", ""))
                    output_path = str(req.get("output_path", ""))
                    if not output_path:
                        raise ValueError("缺少 output_path")
                    if not speaker_wav:
                        raise ValueError("缺少 speaker_wav")

                    if args.stub:
                        sr = 24000
                        wav = np.zeros(int(sr * 0.2), dtype=np.float32)
                    else:
                        prompt = _get_prompt_xvec_only(speaker_wav)
                        wavs, sr = tts.generate_voice_clone(  # type: ignore[union-attr]
                            text=text,
                            language=language,
                            voice_clone_prompt=prompt,
                            max_new_tokens=max_new_tokens,
                        )
                        if not wavs:
                            raise RuntimeError("返回空音频")
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
                    raise ValueError("缺少 items")

                # Collect and validate
                parsed: list[dict] = []
                for idx, it in enumerate(items):
                    if not isinstance(it, dict):
                        raise ValueError(f"items[{idx}] 不是对象")
                    text = str(it.get("text", ""))
                    language = str(it.get("language", "Auto") or "Auto")
                    speaker_wav = str(it.get("speaker_wav", ""))
                    output_path = str(it.get("output_path", ""))
                    if not output_path:
                        raise ValueError(f"items[{idx}] 缺少 output_path")
                    if not speaker_wav:
                        raise ValueError(f"items[{idx}] 缺少 speaker_wav")
                    parsed.append(
                        {
                            "idx": idx,
                            "text": text,
                            "language": language,
                            "speaker_wav": speaker_wav,
                            "output_path": output_path,
                        }
                    )

                results: list[dict] = [{"ok": False, "error": "not processed"} for _ in parsed]

                if args.stub:
                    sr = 24000
                    for it in parsed:
                        idx = int(it["idx"])
                        out = str(it["output_path"])
                        wav = np.zeros(int(sr * 0.2), dtype=np.float32)
                        _write_wav(out, wav, int(sr))
                        results[idx] = {"ok": True, "output_path": out, "sr": int(sr), "n_samples": int(wav.shape[0])}
                    print(json.dumps({"ok": True, "results": results}, ensure_ascii=False), flush=True)
                    continue

                assert tts is not None
                # Group by speaker_wav to maximize prompt reuse.
                by_key: dict[str, list[dict]] = {}
                for it in parsed:
                    key_path = os.path.abspath(str(it["speaker_wav"]))
                    by_key.setdefault(key_path, []).append(it)

                for key_path, group in by_key.items():
                    try:
                        prompt = _get_prompt_xvec_only(key_path)
                        texts = [str(it["text"]) for it in group]
                        langs = [str(it["language"]) for it in group]
                        wavs, sr = tts.generate_voice_clone(  # type: ignore[union-attr]
                            text=texts,
                            language=langs,
                            voice_clone_prompt=prompt,
                            max_new_tokens=max_new_tokens,
                        )
                        if not wavs:
                            raise RuntimeError("返回空音频")
                        if len(wavs) != len(group):
                            raise RuntimeError(f"批量返回数量不匹配: got {len(wavs)}, expected {len(group)}")
                        for wav, it_req in zip(wavs, group):
                            idx = int(it_req["idx"])
                            out = str(it_req["output_path"])
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
                        for it_req in group:
                            idx = int(it_req["idx"])
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

