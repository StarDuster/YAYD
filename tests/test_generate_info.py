"""Generate info 步骤测试."""

from __future__ import annotations

import json
from pathlib import Path


def test_generate_info_creates_video_txt_and_resized_cover(tmp_path: Path):
    from youdub.steps import generate_info as generate_info_fn

    folder = tmp_path / "job"
    folder.mkdir(parents=True, exist_ok=True)

    (folder / "summary.json").write_text(
        json.dumps({"title": "T", "author": "A", "summary": "S", "tags": []}, ensure_ascii=False),
        encoding="utf-8",
    )

    from PIL import Image

    Image.new("RGB", (320, 200), "red").save(folder / "download.png")

    generate_info_fn(str(folder))

    txt = folder / "video.txt"
    cover = folder / "video.png"
    assert txt.exists()
    assert cover.exists()

    assert "T - A" in txt.read_text(encoding="utf-8")

    with Image.open(cover) as img:
        assert img.size == (1280, 960)
