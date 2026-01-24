import json

import pytest

from youdub.config import Settings
from youdub.models import ModelCheckError, ModelManager


def test_merge_segments_respects_speaker_and_punctuation():
    import youdub.core.steps.transcribe as tr

    transcript = [
        {"start": 0.0, "end": 0.5, "text": "hello", "speaker": "S1"},
        {"start": 0.5, "end": 1.0, "text": "world", "speaker": "S1"},
        {"start": 1.0, "end": 1.5, "text": "ok.", "speaker": "S1"},
        {"start": 1.5, "end": 2.0, "text": "next", "speaker": "S1"},
        {"start": 2.0, "end": 3.0, "text": "diff", "speaker": "S2"},
        {"start": 3.0, "end": 4.0, "text": "speaker2", "speaker": "S2"},
    ]

    merged = tr.merge_segments(transcript)
    assert len(merged) == 3

    assert merged[0]["speaker"] == "S1"
    assert merged[0]["start"] == 0.0
    assert merged[0]["end"] == 1.5
    assert merged[0]["text"] == "hello world ok."

    assert merged[1]["speaker"] == "S1"
    assert merged[1]["start"] == 1.5
    assert merged[1]["end"] == 2.0
    assert merged[1]["text"] == "next"

    assert merged[2]["speaker"] == "S2"
    assert merged[2]["start"] == 2.0
    assert merged[2]["end"] == 4.0
    assert merged[2]["text"] == "diff speaker2"


def test_assign_speakers_by_overlap_picks_best_match():
    import youdub.core.steps.transcribe as tr

    segments = [
        {"start": 0.0, "end": 1.0, "text": "a"},
        {"start": 1.0, "end": 2.0, "text": "b"},
    ]
    turns = [
        {"start": 0.2, "end": 1.2, "speaker": "SPEAKER_01"},
        {"start": 1.2, "end": 2.2, "speaker": "SPEAKER_02"},
    ]

    tr._assign_speakers_by_overlap(segments, turns, default_speaker="SPEAKER_00")
    assert segments[0]["speaker"] == "SPEAKER_01"
    assert segments[1]["speaker"] == "SPEAKER_02"


def test_assign_speakers_by_overlap_defaults_when_no_turns_or_bad_segment():
    import youdub.core.steps.transcribe as tr

    segments = [
        {"start": 0.0, "end": 1.0, "text": "a"},
        {"start": 1.0, "end": 1.0, "text": "bad"},  # end <= start
    ]

    tr._assign_speakers_by_overlap(segments, turns=[], default_speaker="SPEAKER_00")
    assert segments[0]["speaker"] == "SPEAKER_00"
    assert segments[1]["speaker"] == "SPEAKER_00"


def test_transcribe_audio_short_circuits_when_transcript_exists(tmp_path):
    import youdub.core.steps.transcribe as tr

    folder = tmp_path / "job"
    folder.mkdir()
    (folder / "transcript.json").write_text(json.dumps([], ensure_ascii=False), encoding="utf-8")

    # Should not require audio file or model availability.
    assert tr.transcribe_audio(str(folder), diarization=False) is True


def test_load_asr_model_requires_model_bin_even_if_dir_nonempty(tmp_path):
    import youdub.core.steps.transcribe as tr

    model_dir = tmp_path / "whisper_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "dummy.txt").write_text("ok", encoding="utf-8")  # make the dir "exist" for ModelManager

    settings = Settings(whisper_model_path=model_dir)
    manager = ModelManager(settings)

    with pytest.raises(ModelCheckError, match=r"model\.bin"):
        tr.load_asr_model(
            model_dir=str(model_dir),
            device="cpu",
            settings=settings,
            model_manager=manager,
            use_batched=False,
        )

