#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import re
import time
from dataclasses import dataclass
from typing import Iterable

import torch

from youdub.config import Settings
from youdub.steps.transcribe import _PYANNOTE_DIARIZATION_MODEL_ID, _find_pyannote_config
from youdub.utils import torch_load_weights_only_compat


@dataclass(frozen=True)
class SrtCue:
    idx: int
    start: float
    end: float
    text: str


_TS_RE = re.compile(r"^(\d{2}):(\d{2}):(\d{2})[,.](\d{3})$")


def _parse_ts(ts: str) -> float:
    m = _TS_RE.match(ts.strip())
    if not m:
        raise ValueError(f"Bad timestamp: {ts!r}")
    hh, mm, ss, ms = (int(x) for x in m.groups())
    return hh * 3600.0 + mm * 60.0 + ss + ms / 1000.0


def parse_srt(path: str) -> list[SrtCue]:
    with open(path, "r", encoding="utf-8") as f:
        raw_lines = [ln.rstrip("\n") for ln in f]

    cues: list[SrtCue] = []
    i = 0
    while i < len(raw_lines):
        line = raw_lines[i].strip()
        if not line:
            i += 1
            continue

        # index line
        try:
            idx = int(line)
        except ValueError:
            # tolerate missing indices: skip until time line
            idx = len(cues) + 1
        i += 1
        if i >= len(raw_lines):
            break

        # time line
        tl = raw_lines[i].strip()
        i += 1
        if "-->" not in tl:
            continue
        left, right = (p.strip() for p in tl.split("-->", 1))
        start = _parse_ts(left)
        end = _parse_ts(right)

        # text lines until blank
        text_lines: list[str] = []
        while i < len(raw_lines) and raw_lines[i].strip():
            text_lines.append(raw_lines[i].strip())
            i += 1
        text = " ".join(text_lines).strip()
        cues.append(SrtCue(idx=idx, start=float(start), end=float(end), text=text))

    # ensure time-ordered
    cues.sort(key=lambda c: (c.start, c.end, c.idx))
    return cues


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _format_float(x: float) -> str:
    if math.isfinite(x):
        return f"{x:.3f}".rstrip("0").rstrip(".")
    return str(x)


def _grid(start: float, stop: float, step: float) -> list[float]:
    # inclusive-ish
    n = int(round((stop - start) / step)) + 1
    out = [start + i * step for i in range(max(1, n))]
    # clamp rounding noise
    return [round(x, 6) for x in out if (x >= start - 1e-9 and x <= stop + 1e-9)]


def _extract_turns(annotation) -> list[dict[str, float | str]]:
    ann_view = getattr(annotation, "exclusive_speaker_diarization", None) or annotation
    turns: list[dict[str, float | str]] = []
    for seg, _, speaker in ann_view.itertracks(yield_label=True):
        turns.append({"start": float(seg.start), "end": float(seg.end), "speaker": str(speaker)})
    turns.sort(key=lambda t: (float(t["start"]), float(t["end"])))
    return turns


@dataclass(frozen=True)
class EvalResult:
    seg_min_off: float
    clust_thr: float
    n_turns: int
    n_speakers: int
    multi_cues: int
    total_cues: int
    flips: int

    @property
    def multi_rate(self) -> float:
        return 0.0 if self.total_cues <= 0 else float(self.multi_cues) / float(self.total_cues)


def evaluate_against_srt(
    cues: list[SrtCue],
    turns: list[dict[str, float | str]],
    *,
    min_secondary_seconds: float = 0.25,
    min_secondary_ratio: float = 0.20,
    flip_gap_seconds: float = 0.70,
) -> tuple[int, int, int, int]:
    """Return (multi_cues, total_cues, flips, n_speakers_assigned)."""

    # pointer for overlapping turns
    t_i = 0
    assigned: list[str] = []
    multi = 0

    for cue in cues:
        c0, c1 = float(cue.start), float(cue.end)
        dur = max(0.0, c1 - c0)

        # advance until turns may overlap
        while t_i < len(turns) and float(turns[t_i]["end"]) <= c0:
            t_i += 1

        overlaps: dict[str, float] = {}
        j = t_i
        while j < len(turns) and float(turns[j]["start"]) < c1:
            spk = str(turns[j]["speaker"])
            o = _overlap(c0, c1, float(turns[j]["start"]), float(turns[j]["end"]))
            if o > 0:
                overlaps[spk] = overlaps.get(spk, 0.0) + o
            j += 1

        if not overlaps:
            # fallback: nearest turn
            best_spk = "SPEAKER_00"
            best_dist = float("inf")
            mid = 0.5 * (c0 + c1)
            for t in turns:
                ts, te = float(t["start"]), float(t["end"])
                d = min(abs(mid - ts), abs(mid - te))
                if d < best_dist:
                    best_dist = d
                    best_spk = str(t["speaker"])
            assigned.append(best_spk)
            continue

        ranked = sorted(overlaps.items(), key=lambda kv: kv[1], reverse=True)
        best_spk, best_o = ranked[0]
        second_o = ranked[1][1] if len(ranked) > 1 else 0.0

        assigned.append(best_spk)

        # "multi-speaker within one cue" heuristic:
        # count only if second speaker occupies a meaningful portion.
        if dur > 1e-6 and second_o >= min_secondary_seconds and second_o >= (min_secondary_ratio * dur):
            multi += 1

    # speaker flips across adjacent cues (when cues are close in time)
    flips = 0
    for i in range(1, len(cues)):
        gap = float(cues[i].start) - float(cues[i - 1].end)
        if gap <= flip_gap_seconds and assigned[i] != assigned[i - 1]:
            flips += 1

    n_speakers_assigned = len(set(assigned))
    return multi, len(cues), flips, n_speakers_assigned


def _combo_iter(
    seg_min_off_values: Iterable[float],
    clust_thr_values: Iterable[float],
) -> list[tuple[float, float]]:
    combos: list[tuple[float, float]] = []
    for s in seg_min_off_values:
        for c in clust_thr_values:
            combos.append((float(s), float(c)))
    return combos


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--folder",
        required=True,
        help="视频文件夹（需要包含 audio_vocals.wav 与 ground_truth.en.srt）",
    )
    ap.add_argument(
        "--wav",
        default="audio_vocals.wav",
        help="相对 folder 的音频文件名（默认 audio_vocals.wav）",
    )
    ap.add_argument(
        "--srt",
        default="ground_truth.en.srt",
        help="相对 folder 的 SRT 文件名（默认 ground_truth.en.srt）",
    )
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--min-speakers", type=int, default=0, help="0 表示不限制")
    ap.add_argument("--max-speakers", type=int, default=0, help="0 表示不限制")

    # coarse grid
    ap.add_argument("--coarse-min-off", default="0,0.3,0.6,1.0")
    ap.add_argument("--coarse-clust", default="0.5,0.6,0.7")

    # refine window
    ap.add_argument("--refine-off-radius", type=float, default=0.2)
    ap.add_argument("--refine-off-step", type=float, default=0.1)
    ap.add_argument("--refine-clust-radius", type=float, default=0.08)
    ap.add_argument("--refine-clust-step", type=float, default=0.02)

    # evaluation knobs
    ap.add_argument("--min-secondary-seconds", type=float, default=0.25)
    ap.add_argument("--min-secondary-ratio", type=float, default=0.20)
    ap.add_argument("--flip-gap-seconds", type=float, default=0.70)

    args = ap.parse_args()

    folder = os.path.abspath(args.folder)
    wav_path = os.path.join(folder, args.wav)
    srt_path = os.path.join(folder, args.srt)

    if not os.path.exists(wav_path):
        raise SystemExit(f"找不到音频: {wav_path}")
    if not os.path.exists(srt_path):
        raise SystemExit(f"找不到字幕: {srt_path}")

    cues = parse_srt(srt_path)
    if not cues:
        raise SystemExit("SRT 为空或解析失败")

    settings = Settings()
    diar_dir = settings.resolve_path(settings.whisper_diarization_model_dir)
    cfg = _find_pyannote_config(diar_dir) if diar_dir else None

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    min_speakers = None if int(args.min_speakers) <= 0 else int(args.min_speakers)
    max_speakers = None if int(args.max_speakers) <= 0 else int(args.max_speakers)

    from pyannote.audio import Pipeline

    with torch_load_weights_only_compat():
        pipeline = Pipeline.from_pretrained(
            str(cfg) if cfg and cfg.exists() else _PYANNOTE_DIARIZATION_MODEL_ID,
            token=settings.hf_token,
        )
    pipeline.to(torch.device(device))

    def run_combo(seg_min_off: float, clust_thr: float) -> EvalResult:
        # instantiate hyperparameters
        pipeline.instantiate(
            {
                "segmentation": {"min_duration_off": float(seg_min_off)},
                "clustering": {"threshold": float(clust_thr)},
            }
        )

        t0 = time.time()
        ann = pipeline(wav_path, min_speakers=min_speakers, max_speakers=max_speakers)
        turns = _extract_turns(ann)

        multi, total, flips, n_spk_assigned = evaluate_against_srt(
            cues,
            turns,
            min_secondary_seconds=float(args.min_secondary_seconds),
            min_secondary_ratio=float(args.min_secondary_ratio),
            flip_gap_seconds=float(args.flip_gap_seconds),
        )
        dt = time.time() - t0
        n_speakers = len({str(t["speaker"]) for t in turns})
        print(
            "trial"
            f" seg.min_off={_format_float(seg_min_off)}"
            f" clust.thr={_format_float(clust_thr)}"
            f" multi={multi}/{total} ({(100.0*multi/total):.2f}%)"
            f" flips={flips}"
            f" spk(turns)={n_speakers}"
            f" spk(cues)={n_spk_assigned}"
            f" turns={len(turns)}"
            f" time={dt:.1f}s"
        )
        return EvalResult(
            seg_min_off=float(seg_min_off),
            clust_thr=float(clust_thr),
            n_turns=len(turns),
            n_speakers=n_speakers,
            multi_cues=multi,
            total_cues=total,
            flips=flips,
        )

    def better(a: EvalResult, b: EvalResult) -> bool:
        # primary: minimize multi-speaker cues
        # secondary: minimize flips, then fewer speakers, then fewer turns
        ka = (a.multi_cues, a.flips, a.n_speakers, a.n_turns)
        kb = (b.multi_cues, b.flips, b.n_speakers, b.n_turns)
        return ka < kb

    def _parse_list(s: str) -> list[float]:
        out: list[float] = []
        for p in (s or "").split(","):
            p = p.strip()
            if not p:
                continue
            out.append(float(p))
        return out

    coarse_off = _parse_list(args.coarse_min_off)
    coarse_thr = _parse_list(args.coarse_clust)

    best: EvalResult | None = None
    print(f"loaded cues={len(cues)} wav={wav_path} device={device}")
    print(f"baseline defaults: seg.min_off={getattr(pipeline.segmentation, 'min_duration_off', None)} clust.thr={getattr(pipeline.clustering, 'threshold', None)}")
    print("== coarse search ==")
    for seg_min_off, clust_thr in _combo_iter(coarse_off, coarse_thr):
        r = run_combo(seg_min_off, clust_thr)
        if best is None or better(r, best):
            best = r
            print(
                "best"
                f" seg.min_off={_format_float(best.seg_min_off)}"
                f" clust.thr={_format_float(best.clust_thr)}"
                f" multi={best.multi_cues}/{best.total_cues}"
                f" flips={best.flips} speakers={best.n_speakers} turns={best.n_turns}"
            )

    assert best is not None
    print("== refine search ==")
    refine_off = _grid(
        max(0.0, best.seg_min_off - float(args.refine_off_radius)),
        best.seg_min_off + float(args.refine_off_radius),
        float(args.refine_off_step),
    )
    refine_thr = _grid(
        max(0.0, best.clust_thr - float(args.refine_clust_radius)),
        min(1.0, best.clust_thr + float(args.refine_clust_radius)),
        float(args.refine_clust_step),
    )

    for seg_min_off, clust_thr in _combo_iter(refine_off, refine_thr):
        r = run_combo(seg_min_off, clust_thr)
        if better(r, best):
            best = r
            print(
                "best"
                f" seg.min_off={_format_float(best.seg_min_off)}"
                f" clust.thr={_format_float(best.clust_thr)}"
                f" multi={best.multi_cues}/{best.total_cues}"
                f" flips={best.flips} speakers={best.n_speakers} turns={best.n_turns}"
            )

    print("== best ==")
    print(
        f"segmentation.min_duration_off={_format_float(best.seg_min_off)}\n"
        f"clustering.threshold={_format_float(best.clust_thr)}\n"
        f"multi_cues={best.multi_cues}/{best.total_cues} ({(100.0*best.multi_rate):.2f}%)\n"
        f"flips={best.flips}\n"
        f"speakers={best.n_speakers}\n"
        f"turns={best.n_turns}\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

