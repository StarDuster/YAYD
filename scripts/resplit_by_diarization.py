#!/usr/bin/env python3
"""按说话人分离结果重新切分 transcript，使用 Whisper word-level timestamps。

目标（面向字幕/后续处理）：
1) 先按说话人聚合（解决同一句话被切成多段的问题）
2) 再用 LLM 补标点（可选）
3) 若句子过长，则按子句尽可能均匀切分
4) 时间戳尽量使用词级时间戳；若 LLM 改动导致无法可靠对齐，则用估算兜底
"""

import json
import math
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from loguru import logger

from youdub.config import Settings
from youdub.steps.transcribe import (
    _find_pyannote_config,
    generate_speaker_audio,
    _PYANNOTE_DIARIZATION_MODEL_ID,
)
from youdub.utils import torch_load_weights_only_compat


def _normalize_ws(text: str) -> str:
    return " ".join((text or "").split()).strip()


_TOKEN_RE = re.compile(r"[0-9A-Za-z_]+(?:['’][0-9A-Za-z_]+)?")


def _tokenize_for_count(text: str) -> list[str]:
    s = _normalize_ws(text).replace("’", "'")
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(s)]


def _word_count(text: str) -> int:
    return len(_tokenize_for_count(text))


def _split_words_to_count(text: str, target_count: int) -> list[str]:
    """兜底：无法按子句切出足够段时，按词尽量均匀切分。"""
    if target_count <= 0:
        return []
    s = _normalize_ws(text)
    if not s:
        return [""] * target_count
    if target_count == 1:
        return [s]
    words = s.split(" ")
    if not words:
        return [""] * target_count

    base = len(words) // target_count
    rem = len(words) % target_count
    out: list[str] = []
    idx = 0
    for i in range(target_count):
        size = base + (1 if i < rem else 0)
        if size <= 0:
            out.append("")
            continue
        out.append(" ".join(words[idx : idx + size]).strip())
        idx += size
    return out


def _group_clauses_evenly(clauses: list[str], target_count: int) -> list[str]:
    """按子句顺序分组，尽量让每组的词数接近。"""
    cleaned = [c.strip() for c in clauses if c and c.strip()]
    if target_count <= 1:
        return [" ".join(cleaned).strip()] if cleaned else []
    if not cleaned:
        return [""] * target_count
    if len(cleaned) < target_count:
        # 子句不够，无法做到“按子句分割到 target_count 段”
        return []

    counts = [_word_count(c) for c in cleaned]
    total = sum(counts) or 1
    target = total / float(target_count)

    out: list[str] = []
    buf: list[str] = []
    buf_cnt = 0

    remaining_groups = target_count
    for i, clause in enumerate(cleaned):
        cnt = counts[i]
        remaining_clauses = len(cleaned) - i

        # 如果剩余子句数 == 剩余组数，则必须一组一个（或先把已有 buf 结算掉）
        if buf and remaining_clauses == remaining_groups:
            out.append(" ".join(buf).strip())
            buf = [clause]
            buf_cnt = cnt
            remaining_groups -= 1
            continue

        if not buf:
            buf = [clause]
            buf_cnt = cnt
            continue

        # 最后一组：把剩余全收进来
        if remaining_groups <= 1:
            buf.append(clause)
            buf_cnt += cnt
            continue

        # 达到目标就收口，否则继续累积
        if buf_cnt >= target:
            out.append(" ".join(buf).strip())
            buf = [clause]
            buf_cnt = cnt
            remaining_groups -= 1
            continue

        # 如果加上当前子句会明显超过目标，而不加又不至于太短，则收口
        if (buf_cnt + cnt) > target and buf_cnt >= max(3, int(target * 0.6)):
            out.append(" ".join(buf).strip())
            buf = [clause]
            buf_cnt = cnt
            remaining_groups -= 1
            continue

        buf.append(clause)
        buf_cnt += cnt

    if buf:
        out.append(" ".join(buf).strip())

    # 极端情况下会偏离目标段数；兜底用词切
    if len(out) != target_count:
        return []
    return out


def _allocate_boundaries(total_words: int, weights: list[int]) -> list[int]:
    """按权重把 total_words 切成 len(weights) 段，返回每段的结束下标（累积，最后=total_words）。"""
    n = len(weights)
    if n <= 0:
        return []
    if total_words <= 0:
        return [0] * n
    wsum = sum(max(0, int(w)) for w in weights)
    if wsum <= 0:
        # 平均切
        return [round((i + 1) * total_words / n) for i in range(n)]

    bounds: list[int] = []
    prev = 0
    cum = 0
    for i, w in enumerate(weights):
        cum += max(0, int(w))
        # 目标边界（四舍五入）
        b = int(round(cum / wsum * total_words))
        # 保证单调且至少推进 1（除非已经到末尾）
        if b <= prev and prev < total_words:
            b = prev + 1
        if b > total_words:
            b = total_words
        bounds.append(b)
        prev = b

    # 最后一段强制对齐到 total_words
    bounds[-1] = total_words
    # 再次修正单调性（可能出现尾部重复）
    for i in range(1, len(bounds)):
        if bounds[i] < bounds[i - 1]:
            bounds[i] = bounds[i - 1]
    return bounds


def _split_punctuated_text_into_parts(punct_text: str, *, max_words: int, max_chars: int) -> list[str]:
    """
    先按句号/问号/感叹号切句；若单句过长则按子句尽量均匀切分。
    返回的 parts 按顺序拼接后应覆盖原文本语义（用于后续对齐/兜底）。
    """
    from youdub.steps.translate import _split_source_text_into_sentences, _split_source_text_relaxed

    parts: list[str] = []
    for sent in [s.strip() for s in _split_source_text_into_sentences(punct_text) if str(s).strip()]:
        s = _normalize_ws(sent)
        if not s:
            continue

        wc = _word_count(s)
        if wc <= 0:
            parts.append(s)
            continue

        if wc <= max_words and len(s) <= max_chars:
            parts.append(s)
            continue

        # 目标段数（至少 2）
        target_count = max(2, int(math.ceil(wc / float(max_words))))
        if max_chars > 0:
            target_count = max(target_count, int(math.ceil(len(s) / float(max_chars))))

        clauses = [c.strip() for c in _split_source_text_relaxed(s) if str(c).strip()]
        grouped = _group_clauses_evenly(clauses, target_count)
        if grouped:
            parts.extend(grouped)
        else:
            # 子句不够/无法均匀 -> 退回按词切
            parts.extend([p for p in _split_words_to_count(s, target_count) if p.strip()])

    return parts


def _punctuate_texts(
    settings: Settings,
    texts: list[str],
    *,
    chunk_size: int = 8,
    attempt_limit: int = 2,
) -> list[str]:
    """复用 translate.py 的 LLM 标点修复逻辑；失败则原文兜底。"""
    from youdub.steps.translate import _punct_fix_chunk

    if not texts:
        return []

    out = list(texts)
    chunk_size = max(1, min(int(chunk_size), 128))
    attempt_limit = max(1, min(int(attempt_limit), 10))

    try:
        for i0 in range(0, len(out), chunk_size):
            idxs = list(range(i0, min(i0 + chunk_size, len(out))))
            payload = {str(i): out[i] for i in idxs}
            fixed, _ok = _punct_fix_chunk(settings, payload, attempt_limit=attempt_limit)
            for i in idxs:
                out[i] = str(fixed.get(str(i), out[i]) or "")
    except Exception as exc:
        # 不因为 LLM 挂了就中断整个后处理
        logger.warning(f"LLM 标点修复失败，将回退到原文（忽略）: {exc}")
        return list(texts)

    return out


def resplit_transcript_by_diarization(
    folder: str,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    device: str = "auto",
    whisper_model: str | None = None,
    punctuate: bool = True,
    max_words_per_sentence: int = 18,
    max_chars_per_sentence: int = 110,
    punct_chunk_size: int = 8,
    punct_attempt_limit: int = 2,
):
    """按 diarization 的 turn 重新切分 transcript。
    
    原理：
    1. 用 Whisper 重新转写，获取精确的 word-level timestamps
    2. 用 pyannote 获取说话人分离结果
    3. 把每个词按其精确时间戳分配到对应的 turn
    """
    
    settings = Settings()
    
    wav_path = os.path.join(folder, "audio_vocals.wav")
    transcript_path = os.path.join(folder, "transcript.json")
    punctuated_path = os.path.join(folder, "transcript_punctuated.json")
    
    if not os.path.exists(wav_path):
        raise FileNotFoundError(f"未找到音频文件: {wav_path}")
    
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # === 第一步：用 Whisper 获取 word-level timestamps ===
    from faster_whisper import WhisperModel
    
    whisper_model_path = whisper_model or str(settings.whisper_model_path)
    compute_type = "float16" if device == "cuda" else "int8"
    
    logger.info(f"加载 Whisper 模型: {whisper_model_path} (设备={device})")
    model = WhisperModel(whisper_model_path, device=device, compute_type=compute_type)
    
    # VAD 参数：使用默认值
    vad_parameters = {
        "min_silence_duration_ms": 2000,  # 默认值
        "speech_pad_ms": 400,             # 默认值
        "threshold": 0.5,                 # 语音概率阈值
    }
    
    logger.info(f"转写并获取词级时间戳: {wav_path}")
    segments, info = model.transcribe(
        wav_path,
        beam_size=5,
        vad_filter=True,
        vad_parameters=vad_parameters,
        word_timestamps=True,  # 关键：启用词级时间戳
        no_speech_threshold=0.8,  # 提高阈值，防止漏句
    )
    
    # 收集所有词及其精确时间戳
    text_segments = []
    for segment in segments:
        if segment.words:
            for word in segment.words:
                text_segments.append({
                    "start": float(word.start),
                    "end": float(word.end),
                    "word": word.word.strip()
                })
    
    logger.info(f"Whisper 转写得到 {len(text_segments)} 个词（带精确时间戳）")
    
    # 释放 Whisper 模型内存
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # === 第二步：运行 pyannote 说话人分离 ===
    from pyannote.audio import Pipeline
    
    diar_dir = settings.resolve_path(settings.whisper_diarization_model_dir)
    token = settings.hf_token
    cfg = _find_pyannote_config(diar_dir) if diar_dir else None
    
    logger.info(f"加载说话人分离管道 (设备={device})")
    
    with torch_load_weights_only_compat():
        if cfg and cfg.exists():
            pipeline = Pipeline.from_pretrained(str(cfg), token=token)
        else:
            pipeline = Pipeline.from_pretrained(_PYANNOTE_DIARIZATION_MODEL_ID, token=token)
    
    pipeline.to(torch.device(device))
    
    logger.info(f"开始说话人分离: {wav_path}")
    ann = pipeline(wav_path, min_speakers=min_speakers, max_speakers=max_speakers)
    
    # 提取 turns
    ann_view = getattr(ann, "exclusive_speaker_diarization", None) or ann
    turns = []
    for seg, _, speaker in ann_view.itertracks(yield_label=True):
        turns.append({
            "start": float(seg.start),
            "end": float(seg.end),
            "speaker": str(speaker)
        })
    
    # 按开始时间排序
    turns.sort(key=lambda x: x["start"])
    
    logger.info(f"检测到 {len(set(t['speaker'] for t in turns))} 个说话人，{len(turns)} 个 turn")
    
    # 为每个词找到最佳匹配的 turn
    # 策略：用词的**结束时间**判断，落在空隙中的词分配给下一个 turn（更符合语义）
    word_to_turn = {}  # word_idx -> turn_idx
    
    for word_idx, word_seg in enumerate(text_segments):
        word_end = word_seg["end"]
        
        best_turn_idx = None
        
        for turn_idx, turn in enumerate(turns):
            turn_start = turn["start"]
            turn_end = turn["end"]
            
            # 词的结束时间落在 turn 范围内
            if turn_start <= word_end <= turn_end:
                best_turn_idx = turn_idx
                break
            
            # 词落在当前 turn 之后、下一个 turn 之前（空隙中）
            # 分配给下一个 turn（即将说话的人）
            if word_end > turn_end:
                next_turn_idx = turn_idx + 1
                if next_turn_idx < len(turns):
                    next_turn_start = turns[next_turn_idx]["start"]
                    # 如果词在空隙中，且更接近下一个 turn
                    if word_end < next_turn_start:
                        # 判断是更接近上一个 turn 的结尾还是下一个 turn 的开头
                        dist_to_prev = word_end - turn_end
                        dist_to_next = next_turn_start - word_end
                        # 优先分配给下一个 turn（因为通常是下一句的开头）
                        best_turn_idx = next_turn_idx
                        break
        
        # 如果还没找到（词在所有 turn 之前或之后），找最近的
        if best_turn_idx is None:
            best_distance = float('inf')
            for turn_idx, turn in enumerate(turns):
                dist = min(abs(word_end - turn["start"]), abs(word_end - turn["end"]))
                if dist < best_distance:
                    best_distance = dist
                    best_turn_idx = turn_idx
        
        if best_turn_idx is not None:
            word_to_turn[word_idx] = best_turn_idx

    # 构建带 speaker 的词序列（按时间顺序）
    words_with_speaker: list[dict[str, float | str]] = []
    missing = 0
    for word_idx, w in enumerate(text_segments):
        turn_idx = word_to_turn.get(word_idx)
        if turn_idx is None:
            missing += 1
            speaker = "SPEAKER_00"
        else:
            speaker = str(turns[turn_idx].get("speaker") or "SPEAKER_00")
        words_with_speaker.append(
            {
                "start": float(w["start"]),
                "end": float(w["end"]),
                "word": str(w["word"]),
                "speaker": speaker,
            }
        )
    if missing:
        logger.warning(f"有 {missing} 个词无法匹配到 turn，已回退到 SPEAKER_00（建议检查 diarization 质量）")

    # === 第三步：说话人聚合（按连续 speaker 合并为 chunk）===
    speaker_chunks: list[dict[str, object]] = []
    cur_speaker: str | None = None
    cur_words: list[dict[str, float | str]] = []

    def _flush_chunk():
        nonlocal cur_speaker, cur_words
        if cur_speaker is None or not cur_words:
            cur_speaker = None
            cur_words = []
            return
        txt = " ".join(str(x["word"]).strip() for x in cur_words if str(x.get("word", "")).strip()).strip()
        if not txt:
            cur_speaker = None
            cur_words = []
            return
        speaker_chunks.append(
            {
                "speaker": cur_speaker,
                "words": list(cur_words),
                "text": txt,
            }
        )
        cur_speaker = None
        cur_words = []

    for w in words_with_speaker:
        spk = str(w.get("speaker") or "SPEAKER_00")
        if cur_speaker is None:
            cur_speaker = spk
            cur_words = [w]
            continue
        if spk != cur_speaker:
            _flush_chunk()
            cur_speaker = spk
            cur_words = [w]
            continue
        cur_words.append(w)
    _flush_chunk()

    logger.info(f"按说话人聚合得到 {len(speaker_chunks)} 个 chunk")

    # === 第四步：LLM 标点（可选）===
    chunk_texts = [str(ch.get("text") or "") for ch in speaker_chunks]
    if punctuate:
        logger.info("开始 LLM 标点修复（用于断句/子句切分）")
        chunk_texts = _punctuate_texts(
            settings,
            chunk_texts,
            chunk_size=punct_chunk_size,
            attempt_limit=punct_attempt_limit,
        )
    else:
        logger.info("跳过 LLM 标点修复（按原文继续）")

    # === 第五步：子句切分 + 词级时间戳对齐（找不到则估算兜底）===
    new_transcript: list[dict[str, object]] = []
    for ch, punct_text in zip(speaker_chunks, chunk_texts, strict=True):
        spk = str(ch.get("speaker") or "SPEAKER_00")
        wlist = ch.get("words")
        if not isinstance(wlist, list) or not wlist:
            continue

        parts = _split_punctuated_text_into_parts(
            str(punct_text or ""),
            max_words=max(1, int(max_words_per_sentence)),
            max_chars=max(0, int(max_chars_per_sentence)),
        )
        parts = [p.strip() for p in parts if str(p).strip()]
        if not parts:
            parts = [str(punct_text or "").strip()] if str(punct_text or "").strip() else []
        if not parts:
            continue

        total_words = len(wlist)
        weights = [max(1, _word_count(p)) for p in parts]

        # 尝试严格对齐：如果 tokens 数量完全相等，就直接按数量切
        strict_total = sum(weights)
        if strict_total != total_words:
            # LLM 改动或 tokenization 不一致：用估算兜底（按权重比例切分）
            logger.debug(
                f"对齐回退（speaker={spk}): words={total_words}, tokens={strict_total} -> 使用比例估算切分"
            )

        boundaries = _allocate_boundaries(total_words, weights)
        prev = 0
        for part_text, b in zip(parts, boundaries, strict=True):
            if b <= prev:
                continue
            seg_words = wlist[prev:b]
            prev = b
            if not seg_words:
                continue

            start_s = float(seg_words[0].get("start", 0.0))  # type: ignore[arg-type]
            end_s = float(seg_words[-1].get("end", start_s))  # type: ignore[arg-type]
            if end_s <= start_s:
                end_s = start_s

            new_transcript.append(
                {
                    "start": start_s,
                    "end": end_s,
                    "text": str(part_text).strip(),
                    "speaker": spk,
                }
            )

    logger.info(f"后处理完成：共 {len(new_transcript)} 个段落（句/子句级）")

    # 统计说话人分布
    speaker_stats: dict[str, int] = {}
    for seg in new_transcript:
        spk = str(seg.get("speaker") or "SPEAKER_00")
        speaker_stats[spk] = speaker_stats.get(spk, 0) + 1

    logger.info("说话人分布:")
    for spk in sorted(speaker_stats.keys()):
        logger.info(f"  {spk}: {speaker_stats[spk]} 段")
    
    # 备份原文件
    backup_path = transcript_path + ".before_resplit.bak"
    if os.path.exists(transcript_path) and not os.path.exists(backup_path):
        import shutil
        shutil.copy(transcript_path, backup_path)
        logger.info(f"已备份原转写到: {backup_path}")
    
    # 保存新转写（已标点、已断句/子句切分）
    with open(transcript_path, "w", encoding="utf-8") as f:
        json.dump(new_transcript, f, indent=2, ensure_ascii=False)
    logger.info(f"已更新转写: {transcript_path}")

    # 同时写入 transcript_punctuated.json 供 translate 复用（长度一致、mtime 更晚则可直接命中缓存）
    try:
        with open(punctuated_path, "w", encoding="utf-8") as f:
            json.dump(new_transcript, f, indent=2, ensure_ascii=False)
        logger.info(f"已写入标点转写缓存: {punctuated_path}")
    except Exception as exc:
        logger.warning(f"写入 {punctuated_path} 失败（忽略）: {exc}")
    
    # 重新生成说话人参考音频
    generate_speaker_audio(folder, new_transcript)
    
    return new_transcript


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="按说话人分离结果重新切分 transcript（使用 Whisper word timestamps）")
    parser.add_argument("folder", help="视频文件夹路径")
    parser.add_argument("--min-speakers", type=int, help="最小说话人数")
    parser.add_argument("--max-speakers", type=int, help="最大说话人数")
    parser.add_argument("--device", default="auto", help="设备 (auto/cuda/cpu)")
    parser.add_argument("--whisper-model", type=str, help="Whisper 模型路径")
    parser.add_argument("--no-punctuate", action="store_true", help="跳过 LLM 标点修复（只做聚合+切分兜底）")
    parser.add_argument("--max-words", type=int, default=18, help="单句最大词数，超过则按子句尽量均匀切分")
    parser.add_argument("--max-chars", type=int, default=110, help="单句最大字符数，超过则按子句尽量均匀切分")
    parser.add_argument("--punct-chunk-size", type=int, default=8, help="LLM 标点修复批大小")
    parser.add_argument("--punct-attempt-limit", type=int, default=2, help="LLM 标点修复重试次数")
    
    args = parser.parse_args()
    
    resplit_transcript_by_diarization(
        args.folder,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        device=args.device,
        whisper_model=args.whisper_model,
        punctuate=(not args.no_punctuate),
        max_words_per_sentence=int(args.max_words),
        max_chars_per_sentence=int(args.max_chars),
        punct_chunk_size=int(args.punct_chunk_size),
        punct_attempt_limit=int(args.punct_attempt_limit),
    )
