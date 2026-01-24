import json
import os
import time
from typing import Iterator

from loguru import logger
from PIL import Image

from ..interrupts import check_cancelled


def resize_thumbnail(folder, size=(1280, 960)):
    image_suffix = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    for suffix in image_suffix:
        image_path = os.path.join(folder, f'download{suffix}')
        if os.path.exists(image_path):
            break
    else:
        return None

    with Image.open(image_path) as img:
        img_ratio = img.width / img.height
        target_ratio = size[0] / size[1]

        if img_ratio < target_ratio:
            new_height = size[1]
            new_width = int(new_height * img_ratio)
        else:
            new_width = size[0]
            new_height = int(new_width / img_ratio)

        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        new_img = Image.new('RGB', size, "black")

        x_offset = (size[0] - new_width) // 2
        y_offset = (size[1] - new_height) // 2
        new_img.paste(img, (x_offset, y_offset))

        new_img_path = os.path.join(folder, 'video.png')
        new_img.save(new_img_path)
        return new_img_path


def generate_summary_txt(folder):
    summary_path = os.path.join(folder, 'summary.json')
    if not os.path.exists(summary_path):
        return

    with open(summary_path, 'r', encoding='utf-8') as f:
        summary = json.load(f)
    
    title = summary.get("title", "Untitled")
    author = summary.get("author", "Unknown")
    summary_text = summary.get("summary", "")
    
    txt = f'{title} - {author}\n\n{summary_text}'
    with open(os.path.join(folder, 'video.txt'), 'w', encoding='utf-8') as f:
        f.write(txt)


def generate_info(folder):
    generate_summary_txt(folder)
    resize_thumbnail(folder)
    

def generate_all_info_under_folder(root_folder):
    check_cancelled()
    start = time.time()
    count = 0
    for root, dirs, files in os.walk(root_folder):
        check_cancelled()
        if 'download.info.json' in files:
            generate_info(root)
            count += 1
    msg = f'Generated all info under {root_folder} (processed {count} folders) in {time.time() - start:.1f}s'
    logger.info(msg)
    return msg


def generate_all_info_under_folder_stream(root_folder: str) -> Iterator[str]:
    """Gradio-friendly streaming version with human-readable progress output."""

    check_cancelled()
    start = time.time()
    root_folder = str(root_folder or "").strip()

    lines: list[str] = []

    def _emit(line: str) -> str:
        # Keep output bounded to avoid extremely large UI payloads.
        lines.append(line)
        if len(lines) > 200:
            del lines[:50]
        return "\n".join(lines)

    if not root_folder:
        yield _emit("错误：Folder 不能为空")
        return
    if not os.path.exists(root_folder):
        yield _emit(f"错误：Folder 不存在：{root_folder}")
        return

    processed = 0
    created_txt = 0
    created_cover = 0
    skipped_no_summary = 0
    skipped_no_cover_src = 0
    errors = 0

    yield _emit(f"开始生成信息：{root_folder}")

    for root, _dirs, files in os.walk(root_folder):
        check_cancelled()
        if "download.info.json" not in files:
            continue

        processed += 1
        yield _emit(f"[{processed}] 处理：{root}")

        # 1) video.txt from summary.json (if present)
        try:
            summary_path = os.path.join(root, "summary.json")
            out_txt_path = os.path.join(root, "video.txt")
            existed_txt = os.path.exists(out_txt_path)
            if os.path.exists(summary_path):
                generate_summary_txt(root)
                if (not existed_txt) and os.path.exists(out_txt_path):
                    created_txt += 1
            else:
                skipped_no_summary += 1
                yield _emit("  - 跳过 video.txt：缺少 summary.json（请先跑“字幕翻译”）")
        except Exception as exc:  # pylint: disable=broad-except
            errors += 1
            logger.exception(f"generate_summary_txt failed: {root} ({exc})")
            yield _emit(f"  - 错误(video.txt)：{type(exc).__name__}: {exc}")

        # 2) video.png cover from download.*
        try:
            out_cover_path = os.path.join(root, "video.png")
            existed_cover = os.path.exists(out_cover_path)
            cover_path = resize_thumbnail(root)
            if cover_path is None:
                skipped_no_cover_src += 1
                yield _emit("  - 跳过封面：未找到 download.(jpg/jpeg/png/bmp/webp)")
            elif (not existed_cover) and os.path.exists(out_cover_path):
                created_cover += 1
        except Exception as exc:  # pylint: disable=broad-except
            errors += 1
            logger.exception(f"resize_thumbnail failed: {root} ({exc})")
            yield _emit(f"  - 错误(video.png)：{type(exc).__name__}: {exc}")

    elapsed = time.time() - start
    done = (
        f"完成：处理 {processed} 个视频目录；新增 video.png={created_cover}，新增 video.txt={created_txt}；"
        f"缺 summary.json 跳过={skipped_no_summary}，缺封面源图跳过={skipped_no_cover_src}；"
        f"错误={errors}；耗时 {elapsed:.1f}s"
    )
    logger.info(done)
    yield _emit(done)

