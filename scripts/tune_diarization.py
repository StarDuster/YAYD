#!/usr/bin/env python3
"""
调优 pyannote 说话人分离参数。
基于 ground_truth.en.srt 的分段结构判断最佳参数组合。
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from loguru import logger

from youdub.config import Settings
from youdub.steps.transcribe import (
    _find_pyannote_config,
    _PYANNOTE_DIARIZATION_MODEL_ID,
)
from youdub.utils import torch_load_weights_only_compat


def parse_srt(srt_path: str) -> list[dict]:
    """解析 SRT 文件，返回 [{start, end, text}, ...]"""
    with open(srt_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    pattern = r"(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n(.*?)(?=\n\n|\Z)"
    matches = re.findall(pattern, content, re.DOTALL)
    
    def parse_time(t: str) -> float:
        h, m, rest = t.split(":")
        s, ms = rest.split(",")
        return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000
    
    segments = []
    for idx, start, end, text in matches:
        segments.append({
            "idx": int(idx),
            "start": parse_time(start),
            "end": parse_time(end),
            "text": text.replace("\n", " ").strip(),
        })
    return segments


def estimate_speaker_changes(segments: list[dict]) -> list[float]:
    """
    基于 SRT 分段估计说话人变化的时间点。
    规则：
    1. 两段之间间隔 > 1.5 秒，可能换人
    2. 内容从第一人称变为第三人称叙述，可能换人
    3. 出现明显的采访问答模式
    """
    change_times = []
    
    narrator_keywords = [
        "data center", "georgia", "facilities", "bills", "week we were",
        "that's patty", "vanderslice says", "we went", "we reached",
        "trump", "ferc", "the two biggest", "their origins",
    ]
    
    interview_keywords = [
        "my cold water", "my front", "we have to", "they destroyed",
        "it's overwhelming", "jeff would", "every month",
    ]
    
    def is_narrator(text: str) -> bool:
        text_lower = text.lower()
        return any(k in text_lower for k in narrator_keywords)
    
    def is_interview(text: str) -> bool:
        text_lower = text.lower()
        return any(k in text_lower for k in interview_keywords)
    
    prev_is_narrator = None
    for i, seg in enumerate(segments):
        if i == 0:
            prev_is_narrator = is_narrator(seg["text"])
            continue
        
        prev_seg = segments[i - 1]
        gap = seg["start"] - prev_seg["end"]
        curr_is_narrator = is_narrator(seg["text"])
        
        # 规则1: 间隔 > 1.5秒
        if gap > 1.5:
            change_times.append(seg["start"])
        # 规则2: 叙述风格变化
        elif prev_is_narrator != curr_is_narrator:
            change_times.append(seg["start"])
        # 规则3: 采访内容变化（简化判断）
        elif is_interview(seg["text"]) != is_interview(prev_seg["text"]):
            change_times.append(seg["start"])
        
        prev_is_narrator = curr_is_narrator
    
    return sorted(set(change_times))


def run_diarization(
    wav_path: str,
    clustering_threshold: float,
    min_duration_off: float,
    segmentation_threshold: float = 0.5,
    device: str = "cuda",
) -> list[dict]:
    """运行 pyannote 分离并返回 turns"""
    from pyannote.audio import Pipeline
    
    settings = Settings()
    diar_dir = settings.resolve_path(settings.whisper_diarization_model_dir)
    token = settings.hf_token
    cfg = _find_pyannote_config(diar_dir) if diar_dir else None
    
    with torch_load_weights_only_compat():
        if cfg and cfg.exists():
            pipeline = Pipeline.from_pretrained(str(cfg), token=token)
        else:
            pipeline = Pipeline.from_pretrained(_PYANNOTE_DIARIZATION_MODEL_ID, token=token)
    
    # 设置参数
    if hasattr(pipeline, "segmentation") and hasattr(pipeline.segmentation, "threshold"):
        pipeline.segmentation.threshold = segmentation_threshold
    if hasattr(pipeline, "segmentation") and hasattr(pipeline.segmentation, "min_duration_off"):
        pipeline.segmentation.min_duration_off = min_duration_off
    if hasattr(pipeline, "clustering") and hasattr(pipeline.clustering, "threshold"):
        pipeline.clustering.threshold = clustering_threshold
    
    pipeline.to(torch.device(device))
    
    ann = pipeline(wav_path)
    ann_view = getattr(ann, "exclusive_speaker_diarization", None) or ann
    
    turns = []
    for seg, _, speaker in ann_view.itertracks(yield_label=True):
        turns.append({
            "start": float(seg.start),
            "end": float(seg.end),
            "speaker": str(speaker),
        })
    return turns


def evaluate_diarization(turns: list[dict], expected_changes: list[float], tolerance: float = 1.0) -> dict:
    """
    评估分离结果：
    - 计算说话人变化点与预期的匹配度
    - 统计说话人数量
    - 计算过度分割率
    """
    # 提取 turn 边界（说话人变化点）
    turn_changes = []
    prev_speaker = None
    for t in turns:
        if prev_speaker is not None and t["speaker"] != prev_speaker:
            turn_changes.append(t["start"])
        prev_speaker = t["speaker"]
    
    # 计算匹配度
    matched = 0
    for expected in expected_changes:
        for actual in turn_changes:
            if abs(expected - actual) < tolerance:
                matched += 1
                break
    
    precision = matched / len(turn_changes) if turn_changes else 0
    recall = matched / len(expected_changes) if expected_changes else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # 统计
    speakers = set(t["speaker"] for t in turns)
    
    return {
        "num_speakers": len(speakers),
        "num_turns": len(turns),
        "num_changes": len(turn_changes),
        "expected_changes": len(expected_changes),
        "matched": matched,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main():
    parser = argparse.ArgumentParser(description="调优 pyannote 说话人分离参数")
    parser.add_argument("folder", help="视频文件夹路径")
    parser.add_argument("--device", default="cuda", help="设备")
    args = parser.parse_args()
    
    folder = args.folder
    srt_path = os.path.join(folder, "ground_truth.en.srt")
    wav_path = os.path.join(folder, "audio_vocals.wav")
    
    if not os.path.exists(srt_path):
        logger.error(f"找不到 SRT 文件: {srt_path}")
        return
    if not os.path.exists(wav_path):
        logger.error(f"找不到音频文件: {wav_path}")
        return
    
    # 解析 SRT 并估计说话人变化点
    segments = parse_srt(srt_path)
    expected_changes = estimate_speaker_changes(segments)
    logger.info(f"从 SRT 估计出 {len(expected_changes)} 个说话人变化点")
    
    # 参数组合
    param_grid = [
        # (clustering_threshold, min_duration_off)
        (0.5, 0.0),   # 宽松聚类，不合并
        (0.5, 0.5),   # 宽松聚类，合并短停顿
        (0.5, 1.0),   # 宽松聚类，合并长停顿
        (0.6, 0.0),   # 中等聚类
        (0.6, 0.5),
        (0.6, 1.0),
        (0.7, 0.0),   # 默认
        (0.7, 0.5),
        (0.7, 1.0),
        (0.8, 0.0),   # 严格聚类
        (0.8, 0.5),
        (0.8, 1.0),
    ]
    
    results = []
    for clustering_threshold, min_duration_off in param_grid:
        logger.info(f"测试参数: clustering={clustering_threshold}, min_duration_off={min_duration_off}")
        try:
            turns = run_diarization(
                wav_path,
                clustering_threshold=clustering_threshold,
                min_duration_off=min_duration_off,
                device=args.device,
            )
            metrics = evaluate_diarization(turns, expected_changes)
            metrics["clustering_threshold"] = clustering_threshold
            metrics["min_duration_off"] = min_duration_off
            results.append(metrics)
            logger.info(f"  -> speakers={metrics['num_speakers']}, turns={metrics['num_turns']}, F1={metrics['f1']:.3f}")
        except Exception as e:
            logger.error(f"  -> 失败: {e}")
    
    # 按 F1 排序
    results.sort(key=lambda x: x["f1"], reverse=True)
    
    print("\n" + "=" * 80)
    print("调参结果（按 F1 分数排序）")
    print("=" * 80)
    for r in results:
        print(f"clustering={r['clustering_threshold']:.1f}, min_off={r['min_duration_off']:.1f} | "
              f"speakers={r['num_speakers']:2d}, turns={r['num_turns']:3d}, "
              f"P={r['precision']:.3f}, R={r['recall']:.3f}, F1={r['f1']:.3f}")
    
    if results:
        best = results[0]
        print("\n" + "=" * 80)
        print(f"推荐参数: clustering_threshold={best['clustering_threshold']}, "
              f"min_duration_off={best['min_duration_off']}")
        print("=" * 80)


if __name__ == "__main__":
    main()
