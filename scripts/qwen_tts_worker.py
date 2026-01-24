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
    xvec_prompt_cache: dict[str, object] = {}
    spk_embedding_cache: dict[str, object] = {}
    VoiceClonePromptItem = None

    if not args.stub:
        import torch
        from qwen_tts import Qwen3TTSModel
        from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem as _VoiceClonePromptItem

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
        VoiceClonePromptItem = _VoiceClonePromptItem

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

    def _get_speaker_embedding(anchor_wav: str):
        assert tts is not None
        key = os.path.abspath(anchor_wav)
        emb = spk_embedding_cache.get(key)
        if emb is None:
            prompt = _get_prompt_xvec_only(anchor_wav)
            try:
                emb = prompt[0].ref_spk_embedding  # type: ignore[attr-defined]
            except Exception as exc:
                raise RuntimeError("无法从 voice clone prompt 提取 ref_spk_embedding") from exc
            spk_embedding_cache[key] = emb
        return emb

    def _to_bool(v, default: bool = False) -> bool:
        if v is None:
            return bool(default)
        if isinstance(v, bool):
            return v
        s = str(v).strip().lower()
        if s in {"1", "true", "yes", "y", "on"}:
            return True
        if s in {"0", "false", "no", "n", "off"}:
            return False
        return bool(default)

    def _build_prompt_hybrid(
        speaker_anchor_wav: str,
        icl_audio: str,
        icl_text: str,
    ):
        """Hybrid prompt: speaker embedding from anchor, prosody from (icl_audio, icl_text)."""
        assert tts is not None
        if VoiceClonePromptItem is None:
            raise RuntimeError("VoiceClonePromptItem 未加载")
        if not speaker_anchor_wav:
            speaker_anchor_wav = icl_audio
        spk_emb = _get_speaker_embedding(speaker_anchor_wav)
        icl_items = tts.create_voice_clone_prompt(
            ref_audio=icl_audio,
            ref_text=icl_text,
            x_vector_only_mode=False,
        )
        if not icl_items:
            raise RuntimeError("ICL prompt 为空")
        out = []
        for it in icl_items:
            out.append(
                VoiceClonePromptItem(
                    ref_code=getattr(it, "ref_code", None),
                    ref_spk_embedding=spk_emb,
                    x_vector_only_mode=False,
                    icl_mode=True,
                    ref_text=getattr(it, "ref_text", None) or icl_text,
                )
            )
        return out

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
                    speaker_anchor_wav = str(req.get("speaker_anchor_wav", "") or "")
                    icl_audio = str(req.get("icl_audio", "") or "")
                    icl_text = str(req.get("icl_text", "") or "")
                    xvec_only = _to_bool(req.get("x_vector_only_mode"), default=(not (icl_audio and icl_text)))
                    output_path = str(req.get("output_path", ""))
                    if not output_path:
                        raise ValueError("缺少 output_path")
                    if not speaker_anchor_wav:
                        speaker_anchor_wav = speaker_wav
                    if not speaker_wav and not speaker_anchor_wav:
                        raise ValueError("缺少 speaker_wav/speaker_anchor_wav")

                    if args.stub:
                        sr = 24000
                        wav = np.zeros(int(sr * 0.2), dtype=np.float32)
                    else:
                        if (not xvec_only) and icl_audio and icl_text:
                            prompt = _build_prompt_hybrid(speaker_anchor_wav, icl_audio, icl_text)
                        else:
                            prompt = _get_prompt_xvec_only(speaker_wav or speaker_anchor_wav)
                        wavs, sr = tts.generate_voice_clone(  # type: ignore[union-attr]
                            text=text,
                            language=language,
                            voice_clone_prompt=prompt,
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
                    speaker_anchor_wav = str(it.get("speaker_anchor_wav", "") or "")
                    icl_audio = str(it.get("icl_audio", "") or "")
                    icl_text = str(it.get("icl_text", "") or "")
                    xvec_only = _to_bool(it.get("x_vector_only_mode"), default=(not (icl_audio and icl_text)))
                    output_path = str(it.get("output_path", ""))
                    if not output_path:
                        raise ValueError(f"items[{idx}] 缺少 output_path")
                    if not speaker_anchor_wav:
                        speaker_anchor_wav = speaker_wav
                    if not speaker_wav and not speaker_anchor_wav:
                        raise ValueError(f"items[{idx}] 缺少 speaker_wav/speaker_anchor_wav")
                    parsed.append(
                        {
                            "idx": idx,
                            "text": text,
                            "language": language,
                            "speaker_wav": speaker_wav,
                            "speaker_anchor_wav": speaker_anchor_wav,
                            "icl_audio": icl_audio,
                            "icl_text": icl_text,
                            "x_vector_only_mode": bool(xvec_only),
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
                # Group by mode + speaker source to maximize prompt reuse.
                by_key: dict[tuple[str, str], list[dict]] = {}
                for it in parsed:
                    mode = "icl" if (not it["x_vector_only_mode"] and it["icl_audio"] and it["icl_text"]) else "xvec"
                    if mode == "icl":
                        k = ("icl", os.path.abspath(str(it["speaker_anchor_wav"] or it["speaker_wav"])))
                    else:
                        k = ("xvec", os.path.abspath(str(it["speaker_wav"] or it["speaker_anchor_wav"])))
                    by_key.setdefault(k, []).append(it)

                for (mode, key_path), group in by_key.items():
                    try:
                        if mode == "icl":
                            if VoiceClonePromptItem is None:
                                raise RuntimeError("VoiceClonePromptItem 未加载")
                            anchor = key_path
                            spk_emb = _get_speaker_embedding(anchor)
                            texts = [str(it["text"]) for it in group]
                            langs = [str(it["language"]) for it in group]
                            icl_audios = [str(it["icl_audio"]) for it in group]
                            icl_texts = [str(it["icl_text"]) for it in group]
                            icl_items = tts.create_voice_clone_prompt(
                                ref_audio=icl_audios,
                                ref_text=icl_texts,
                                x_vector_only_mode=False,
                            )
                            if len(icl_items) != len(group):
                                raise RuntimeError(
                                    f"ICL prompt 数量不匹配: got {len(icl_items)}, expected {len(group)}"
                                )
                            prompt_items = []
                            for it_prompt, it_req in zip(icl_items, group):
                                prompt_items.append(
                                    VoiceClonePromptItem(
                                        ref_code=getattr(it_prompt, "ref_code", None),
                                        ref_spk_embedding=spk_emb,
                                        x_vector_only_mode=False,
                                        icl_mode=True,
                                        ref_text=getattr(it_prompt, "ref_text", None) or str(it_req["icl_text"]),
                                    )
                                )
                            wavs, sr = tts.generate_voice_clone(  # type: ignore[union-attr]
                                text=texts,
                                language=langs,
                                voice_clone_prompt=prompt_items,
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
                        else:
                            speaker_wav = key_path
                            prompt = _get_prompt_xvec_only(speaker_wav)
                            texts = [str(it["text"]) for it in group]
                            langs = [str(it["language"]) for it in group]
                            wavs, sr = tts.generate_voice_clone(  # type: ignore[union-attr]
                                text=texts,
                                language=langs,
                                voice_clone_prompt=prompt,
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

