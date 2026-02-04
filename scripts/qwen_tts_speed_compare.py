#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


READY_LINE = "__READY__"

DEFAULT_BASELINE_SPEED = 1.1
DEFAULT_CLAMP_GAP_THRESHOLD = 1.7

_EN_SENT_END = {".", "!", "?"}
_ZH_SENT_END = {"。", "！", "？"}
_TRAILING_CLOSERS = "\"'”’)]}）】》"


def _safe_float(v: Any, default: float | None = None) -> float | None:
    try:
        x = float(v)
    except Exception:
        return default
    if not (x == x and abs(x) != float("inf")):
        return default
    return x


def _duration_from_worker_result(r: dict) -> float | None:
    try:
        sr = r.get("sr")
        n = r.get("n_samples")
        sr_f = float(sr)
        n_f = float(n)
        if not (sr_f > 0.0 and n_f >= 0.0):
            return None
        return float(n_f / sr_f)
    except Exception:
        return None


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _strip_trailing_closers(s: str) -> str:
    t = str(s or "")
    while t and t[-1] in _TRAILING_CLOSERS:
        t = t[:-1]
    return t


def _endswith_sentence_punct_en(s: str) -> bool:
    t = _strip_trailing_closers(str(s or "").strip())
    if not t:
        return False
    # Treat ellipsis as "continuation", not a boundary.
    if t.endswith("...") or t.endswith("…"):
        return False
    return t[-1] in _EN_SENT_END


def _pick_zh_punct_for_en(en_text: str) -> str:
    t = _strip_trailing_closers(str(en_text or "").strip())
    if not t:
        return "。"
    if t.endswith("?"):
        return "？"
    if t.endswith("!"):
        return "！"
    return "。"


def _ensure_sentence_end_en(s: str) -> str:
    t = str(s or "").strip()
    if not t:
        return ""
    # Note: treat ellipsis as terminal for "ensure", but NOT as a sentence boundary.
    if _endswith_sentence_punct_en(t) or t.endswith("...") or t.endswith("…"):
        return t
    return t + "."


def _ensure_sentence_end_zh(s: str, *, prefer: str) -> str:
    t = str(s or "").strip()
    if not t:
        return ""
    t2 = _strip_trailing_closers(t)
    if t2 and (t2.endswith("...") or t2.endswith("…") or t2.endswith("……") or t2[-1] in _ZH_SENT_END):
        return t
    punct = prefer if prefer in _ZH_SENT_END else "。"
    return t + punct


def _join_en(parts: list[str]) -> str:
    out = ""
    for p in parts:
        s = str(p or "").strip()
        if not s:
            continue
        if not out:
            out = s
            continue
        if out.endswith(("-", "—", "–")):
            out = out + s
        else:
            out = out + " " + s
    return out.strip()


def _join_zh(parts: list[str]) -> str:
    out = ""
    for p in parts:
        s = str(p or "").strip()
        if not s:
            continue
        out = out + s
    return out.strip()


def _clamp_gap(raw_ratio: float, applied_ratio: float) -> float:
    raw = float(raw_ratio)
    applied = float(applied_ratio)
    if not (raw > 0.0 and applied > 0.0):
        return float("inf")
    a = raw / applied
    b = applied / raw
    return float(max(a, b))


def _build_sentence_units(
    segments: list[dict[str, Any]],
    *,
    max_segs_per_sentence: int = 12,
    max_chars_per_sentence: int = 500,
) -> list[dict[str, Any]]:
    """
    Build "sentence units" by concatenating consecutive segments.

    Rules (simple + stable):
    - Prefer breaking when English ends with sentence punctuation (.!?).
    - Hard break on max segments / max chars to avoid runaway.
    - Ensure both EN and ZH end with sentence-ending punctuation before sending to TTS.
    """
    units: list[dict[str, Any]] = []
    cur_indices: list[int] = []
    cur_en_parts: list[str] = []
    cur_zh_parts: list[str] = []
    cur_speaker: str | None = None
    cur_start: float | None = None
    cur_end: float | None = None

    def _flush() -> None:
        nonlocal cur_indices, cur_en_parts, cur_zh_parts, cur_speaker, cur_start, cur_end
        if not cur_indices:
            return
        en_raw = _join_en(cur_en_parts)
        en_text = _ensure_sentence_end_en(en_raw)
        zh_raw = _join_zh(cur_zh_parts)
        zh_text = _ensure_sentence_end_zh(zh_raw, prefer=_pick_zh_punct_for_en(en_text))
        units.append(
            {
                "index": int(len(units)),
                "segment_indices": list(cur_indices),
                "start": cur_start,
                "end": cur_end,
                "speaker": str(cur_speaker or "SPEAKER_00"),
                "text": en_text,
                "translation": zh_text,
            }
        )
        cur_indices = []
        cur_en_parts = []
        cur_zh_parts = []
        cur_speaker = None
        cur_start = None
        cur_end = None

    max_segs = int(max(1, max_segs_per_sentence))
    max_chars = int(max(50, max_chars_per_sentence))

    for i, seg in enumerate(segments):
        speaker = str(seg.get("speaker") or "SPEAKER_00")
        en = str(seg.get("text") or "").strip()
        zh = str(seg.get("translation") or "").strip()

        if not cur_indices:
            cur_speaker = speaker
            cur_start = _safe_float(seg.get("start"), default=None)

        cur_indices.append(int(i))
        cur_en_parts.append(en)
        cur_zh_parts.append(zh)
        cur_end = _safe_float(seg.get("end"), default=None)

        en_now = _join_en(cur_en_parts)
        hard_break = (len(cur_indices) >= max_segs) or (len(en_now) >= max_chars)
        if hard_break or _endswith_sentence_punct_en(en_now):
            _flush()

    _flush()
    return units


@dataclass
class WorkerItem:
    idx: int
    text: str
    language: str
    speaker_wav: str
    output_path: str

    def to_req(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "language": self.language,
            "speaker_wav": self.speaker_wav,
            "output_path": self.output_path,
        }


class QwenWorkerClient:
    def __init__(
        self,
        *,
        python_exe: str,
        worker_script: str,
        model_path: str,
        startup_timeout_sec: float = 1800.0,
    ) -> None:
        self._proc = subprocess.Popen(  # noqa: S603
            [python_exe, "-u", worker_script, "--model-path", model_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        assert self._proc.stdin is not None
        assert self._proc.stdout is not None

        self._stderr_tail: list[str] = []
        self._stderr_q: queue.Queue[str] = queue.Queue()

        def _drain_stderr() -> None:
            assert self._proc.stderr is not None
            try:
                for line in self._proc.stderr:
                    s = (line or "").rstrip("\n")
                    if not s:
                        continue
                    self._stderr_q.put(s)
            except Exception:
                return

        threading.Thread(target=_drain_stderr, daemon=True).start()

        begin = time.monotonic()
        while True:
            self._pull_stderr_tail(max_lines=200)
            if self._proc.poll() is not None:
                raise RuntimeError(self._format_startup_error("worker exited early"))
            if time.monotonic() - begin > float(startup_timeout_sec):
                self.close()
                raise RuntimeError(self._format_startup_error(f"startup timeout ({startup_timeout_sec:.0f}s)"))
            line = self._proc.stdout.readline()
            if not line:
                time.sleep(0.05)
                continue
            s = line.strip()
            if not s:
                continue
            if s == READY_LINE:
                break

    def _pull_stderr_tail(self, *, max_lines: int = 200) -> None:
        try:
            while True:
                s = self._stderr_q.get_nowait()
                self._stderr_tail.append(s)
                if len(self._stderr_tail) > max_lines:
                    self._stderr_tail = self._stderr_tail[-max_lines:]
        except queue.Empty:
            return

    def _format_startup_error(self, reason: str) -> str:
        self._pull_stderr_tail(max_lines=200)
        tail = "\n".join(self._stderr_tail[-80:]).strip()
        extra = f"\nstderr_tail:\n{tail}" if tail else ""
        rc = self._proc.poll()
        return f"Qwen worker start failed: {reason}, exit_code={rc}{extra}"

    def close(self) -> None:
        try:
            if self._proc.poll() is None:
                try:
                    assert self._proc.stdin is not None
                    self._proc.stdin.write(json.dumps({"cmd": "shutdown"}) + "\n")
                    self._proc.stdin.flush()
                except Exception:
                    pass
                try:
                    self._proc.terminate()
                except Exception:
                    pass
        finally:
            try:
                self._proc.kill()
            except Exception:
                pass

    def synthesize_batch(self, items: list[WorkerItem], *, timeout_sec: float = 600.0) -> list[dict[str, Any]]:
        if not items:
            return []
        if self._proc.poll() is not None:
            raise RuntimeError("Qwen worker exited")
        assert self._proc.stdin is not None
        assert self._proc.stdout is not None
        self._pull_stderr_tail(max_lines=200)

        req = {"cmd": "synthesize_batch", "items": [it.to_req() for it in items]}
        self._proc.stdin.write(json.dumps(req, ensure_ascii=False) + "\n")
        self._proc.stdin.flush()

        begin = time.monotonic()
        skipped: list[str] = []
        while True:
            self._pull_stderr_tail(max_lines=200)
            if time.monotonic() - begin > float(timeout_sec):
                tail = "\n".join(self._stderr_tail[-80:]).strip()
                extra = f"\nstderr_tail:\n{tail}" if tail else ""
                raise RuntimeError(f"Qwen worker batch timeout ({timeout_sec:.0f}s){extra}")
            line = self._proc.stdout.readline()
            if not line:
                time.sleep(0.05)
                continue
            s = line.strip()
            if not s:
                continue
            try:
                resp = json.loads(s)
            except Exception:
                skipped.append(s)
                if len(skipped) > 30:
                    skipped = skipped[-30:]
                continue
            if not isinstance(resp, dict) or not resp.get("ok"):
                err = ""
                if isinstance(resp, dict):
                    err = str(resp.get("error", "unknown error"))
                tail = "\n".join(self._stderr_tail[-80:]).strip()
                extra = f"\nstderr_tail:\n{tail}" if tail else ""
                raise RuntimeError(f"Qwen worker batch failed: {err}{extra}")
            results = resp.get("results")
            if not isinstance(results, list) or len(results) != len(items):
                raise RuntimeError(
                    f"Unexpected batch results: type={type(results)} len={getattr(results, '__len__', lambda: -1)()}"
                )
            out: list[dict[str, Any]] = []
            for r in results:
                out.append(r if isinstance(r, dict) else {"ok": False, "error": "invalid result item"})
            return out


def _find_job_dirs(root: Path) -> list[Path]:
    out: list[Path] = []
    for p in root.rglob("translation.json"):
        if p.is_file():
            out.append(p.parent)
    return sorted(set(out))


def _pick_speaker_wav(job_dir: Path, speaker: str) -> Path | None:
    spk_dir = job_dir / "SPEAKER"
    cand = spk_dir / f"{speaker}.wav"
    if cand.exists():
        return cand
    fallback = spk_dir / "SPEAKER_00.wav"
    if fallback.exists():
        return fallback
    if spk_dir.exists():
        wavs = sorted([p for p in spk_dir.glob("*.wav") if p.is_file()])
        if wavs:
            return wavs[0]
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Batch synthesize EN/ZH with Qwen3-TTS and compute duration ratios.")
    ap.add_argument(
        "--root",
        default=str(Path("videos") / "More Perfect Union"),
        help="Root folder containing one or more job subfolders with translation.json",
    )
    ap.add_argument("--batch-size", type=int, default=8, help="Batch size (number of items per request).")
    ap.add_argument(
        "--unit",
        choices=["segment", "sentence"],
        default="segment",
        help="Synthesis unit: raw segments or merged whole sentences.",
    )
    ap.add_argument("--max-segs-per-sentence", type=int, default=12, help="Max segments to merge into one sentence unit.")
    ap.add_argument("--max-chars-per-sentence", type=int, default=500, help="Max EN chars to merge into one sentence unit.")
    ap.add_argument("--limit", type=int, default=0, help="Optional limit of segments per job (0 = all).")
    ap.add_argument("--language-en", default="English", help="Language tag passed to Qwen TTS for English text.")
    ap.add_argument("--language-zh", default="Auto", help="Language tag passed to Qwen TTS for Chinese text.")
    ap.add_argument("--out-dirname", default="qwen_speed_compare", help="Output subfolder name under each job.")
    ap.add_argument(
        "--baseline-speed",
        type=float,
        default=float(DEFAULT_BASELINE_SPEED),
        help="Baseline playback speed-up factor for ZH (e.g. 1.1 => duration ratio 1/1.1). Used for outlier checks.",
    )
    ap.add_argument(
        "--clamp-gap-threshold",
        type=float,
        default=float(DEFAULT_CLAMP_GAP_THRESHOLD),
        help="Outlier threshold on clamp gap between per-unit required ratio and baseline ratio.",
    )
    ap.add_argument(
        "--report-name",
        default="qwen_tts_speed_compare.json",
        help="Report JSON filename under each job output folder.",
    )
    args = ap.parse_args()

    root = Path(args.root).expanduser().absolute()
    if not root.exists():
        raise SystemExit(f"Root not found: {root}")

    batch_size = int(max(1, min(int(args.batch_size or 8), 64)))
    limit = int(max(0, int(args.limit or 0)))
    lang_en = str(args.language_en or "English")
    lang_zh = str(args.language_zh or "Auto")
    unit_mode = str(args.unit or "segment").strip().lower()
    if unit_mode not in {"segment", "sentence"}:
        unit_mode = "segment"

    baseline_speed = _safe_float(args.baseline_speed, default=float(DEFAULT_BASELINE_SPEED)) or float(DEFAULT_BASELINE_SPEED)
    if not (baseline_speed > 0.0):
        baseline_speed = float(DEFAULT_BASELINE_SPEED)
    baseline_ratio = float(1.0 / float(baseline_speed))

    clamp_gap_threshold = _safe_float(args.clamp_gap_threshold, default=float(DEFAULT_CLAMP_GAP_THRESHOLD)) or float(
        DEFAULT_CLAMP_GAP_THRESHOLD
    )
    if not (clamp_gap_threshold > 1.0):
        clamp_gap_threshold = float(DEFAULT_CLAMP_GAP_THRESHOLD)

    repo_root = Path(__file__).resolve().parents[1]
    worker_script = str((repo_root / "scripts" / "qwen_tts_worker.py").absolute())
    model_dir = os.getenv(
        "QWEN_TTS_MODEL_PATH",
        str((repo_root / "models" / "TTS" / "Qwen3-TTS-12Hz-1.7B-Base").absolute()),
    )
    model_dir = str(Path(model_dir).expanduser().absolute())
    python_exe = os.getenv("QWEN_TTS_PYTHON", sys.executable)
    python_exe = str(Path(python_exe).expanduser().absolute())

    job_dirs = _find_job_dirs(root)
    if not job_dirs:
        raise SystemExit(f"No jobs found under: {root} (expected **/translation.json)")

    client = QwenWorkerClient(python_exe=python_exe, worker_script=worker_script, model_path=model_dir)
    try:
        for job_dir in job_dirs:
            trans_path = job_dir / "translation.json"
            segments = _load_json(trans_path)
            if not isinstance(segments, list) or not segments:
                continue
            if limit > 0:
                segments = segments[:limit]

            if unit_mode == "sentence":
                units = _build_sentence_units(
                    segments,
                    max_segs_per_sentence=int(args.max_segs_per_sentence or 12),
                    max_chars_per_sentence=int(args.max_chars_per_sentence or 500),
                )
            else:
                units = [
                    {
                        "index": int(i),
                        "segment_indices": [int(i)],
                        "start": _safe_float(seg.get("start"), default=None),
                        "end": _safe_float(seg.get("end"), default=None),
                        "speaker": str(seg.get("speaker") or "SPEAKER_00"),
                        "text": str(seg.get("text") or "").strip(),
                        "translation": str(seg.get("translation") or "").strip(),
                    }
                    for i, seg in enumerate(segments)
                ]

            out_root = job_dir / str(args.out_dirname)
            out_unit_root = out_root if unit_mode == "segment" else (out_root / unit_mode)
            out_en = out_unit_root / "en"
            out_zh = out_unit_root / "zh"
            out_en.mkdir(parents=True, exist_ok=True)
            out_zh.mkdir(parents=True, exist_ok=True)

            seg_results: list[dict[str, Any]] = []
            total_en = 0.0
            total_zh = 0.0
            total_zh_baseline = 0.0
            ok_pairs = 0
            failed = 0
            outliers = 0

            def _run_language(lang: str, *, key: str) -> dict[int, dict[str, Any]]:
                dur_by_idx: dict[int, dict[str, Any]] = {}
                batch: list[WorkerItem] = []
                for u in units:
                    i = int(u.get("index", -1))
                    text = str(u.get(key) or "").strip()
                    if not text:
                        continue
                    speaker = str(u.get("speaker") or "SPEAKER_00")
                    spk_wav = _pick_speaker_wav(job_dir, speaker)
                    if spk_wav is None:
                        continue
                    out_dir = out_en if key == "text" else out_zh
                    out_path = out_dir / f"{i:04d}.wav"
                    batch.append(
                        WorkerItem(
                            idx=i,
                            text=text,
                            language=lang,
                            speaker_wav=str(spk_wav),
                            output_path=str(out_path),
                        )
                    )
                    if len(batch) >= batch_size:
                        res = client.synthesize_batch(batch, timeout_sec=600.0)
                        for it, rr in zip(batch, res):
                            dur_by_idx[it.idx] = rr
                        batch = []
                if batch:
                    res = client.synthesize_batch(batch, timeout_sec=600.0)
                    for it, rr in zip(batch, res):
                        dur_by_idx[it.idx] = rr
                return dur_by_idx

            # Note: use separate passes so "bs=8" means 8 items per language batch.
            en_res = _run_language(lang_en, key="text")
            zh_res = _run_language(lang_zh, key="translation")

            for u in units:
                i = int(u.get("index", -1))
                en_r = en_res.get(i) or {}
                zh_r = zh_res.get(i) or {}
                dur_en = _duration_from_worker_result(en_r) if en_r.get("ok") else None
                dur_zh = _duration_from_worker_result(zh_r) if zh_r.get("ok") else None

                if dur_en is not None and dur_zh is not None and dur_en > 0 and dur_zh > 0:
                    ratio_dur_en_over_zh = float(dur_en / dur_zh)  # required (new/old) to match EN duration
                    ratio_dur_zh_over_en = float(dur_zh / dur_en)
                    total_en += float(dur_en)
                    total_zh += float(dur_zh)
                    total_zh_baseline += float(dur_zh * baseline_ratio)
                    ok_pairs += 1
                else:
                    ratio_dur_en_over_zh = None
                    ratio_dur_zh_over_en = None
                    if str(u.get("text") or "").strip() or str(u.get("translation") or "").strip():
                        failed += 1

                if ratio_dur_en_over_zh is not None:
                    clamp_gap_to_baseline = float(_clamp_gap(float(ratio_dur_en_over_zh), float(baseline_ratio)))
                    is_outlier = bool(clamp_gap_to_baseline > float(clamp_gap_threshold) + 1e-9)
                else:
                    clamp_gap_to_baseline = None
                    is_outlier = False

                if is_outlier:
                    outliers += 1

                dur_zh_after_baseline = (
                    float(dur_zh * baseline_ratio)
                    if (dur_zh is not None and dur_zh > 0.0 and baseline_ratio > 0.0)
                    else None
                )
                ratio_after_baseline_zh_over_en = (
                    float(dur_zh_after_baseline / dur_en)
                    if (
                        dur_en is not None
                        and dur_en > 0.0
                        and dur_zh_after_baseline is not None
                        and dur_zh_after_baseline > 0.0
                    )
                    else None
                )

                seg_results.append(
                    {
                        "index": i,
                        "unit": unit_mode,
                        "segment_indices": list(u.get("segment_indices") or []),
                        "start": _safe_float(u.get("start"), default=None),
                        "end": _safe_float(u.get("end"), default=None),
                        "speaker": str(u.get("speaker") or ""),
                        "text_en": str(u.get("text") or ""),
                        "text_zh": str(u.get("translation") or ""),
                        "qwen_language_en": lang_en,
                        "qwen_language_zh": lang_zh,
                        "dur_en_sec": dur_en,
                        "dur_zh_sec": dur_zh,
                        "dur_zh_after_baseline_sec": dur_zh_after_baseline,
                        "baseline_speed_zh": float(baseline_speed),
                        "baseline_ratio_new_over_old": float(baseline_ratio),
                        "ratio_duration_en_over_zh": ratio_dur_en_over_zh,
                        "ratio_duration_zh_over_en": ratio_dur_zh_over_en,
                        "ratio_after_baseline_zh_over_en": ratio_after_baseline_zh_over_en,
                        "clamp_gap_to_baseline": clamp_gap_to_baseline,
                        "is_outlier_to_baseline": bool(is_outlier),
                        "en_worker": en_r,
                        "zh_worker": zh_r,
                    }
                )

            overall: dict[str, Any] = {
                "segments_total": int(len(segments)),
                "units_total": int(len(units)),
                "pairs_ok": int(ok_pairs),
                "pairs_failed_or_missing": int(failed),
                "outliers_to_baseline": int(outliers),
                "sum_dur_en_sec": float(total_en),
                "sum_dur_zh_sec": float(total_zh),
                "sum_dur_zh_after_baseline_sec": float(total_zh_baseline),
                "baseline_speed_zh": float(baseline_speed),
                "baseline_ratio_new_over_old": float(baseline_ratio),
                "overall_ratio_duration_en_over_zh": (float(total_en / total_zh) if total_en > 0 and total_zh > 0 else None),
                "overall_ratio_duration_zh_over_en": (float(total_zh / total_en) if total_en > 0 and total_zh > 0 else None),
                "overall_ratio_after_baseline_zh_over_en": (
                    float(total_zh_baseline / total_en) if total_en > 0 and total_zh_baseline > 0 else None
                ),
                "overall_clamp_gap_to_baseline": (
                    float(_clamp_gap(float(total_en / total_zh), float(baseline_ratio)))
                    if total_en > 0 and total_zh > 0
                    else None
                ),
            }

            report_name = str(args.report_name)
            if report_name == "qwen_tts_speed_compare.json" and unit_mode == "sentence":
                report_name = "qwen_tts_speed_compare_sentence.json"

            report = {
                "job_dir": str(job_dir),
                "translation_json": str(trans_path),
                "batch_size": int(batch_size),
                "unit": unit_mode,
                "python_exe": python_exe,
                "worker_script": worker_script,
                "model_path": model_dir,
                "language_en": lang_en,
                "language_zh": lang_zh,
                "baseline_speed_zh": float(baseline_speed),
                "baseline_ratio_new_over_old": float(baseline_ratio),
                "clamp_gap_threshold": float(clamp_gap_threshold),
                "overall": overall,
                "segments": seg_results,
            }

            _dump_json(job_dir / report_name, report)
            _dump_json(out_unit_root / "overall.json", overall)
    finally:
        try:
            client.close()
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

