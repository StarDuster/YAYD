import os
import re
from typing import Any, Generator

import yt_dlp
from loguru import logger

from ..interrupts import check_cancelled


def sanitize_title(title: str) -> str:
    """Sanitize the video title for file system usage."""
    title = re.sub(r'[^\w\u4e00-\u9fff \d_-]', '', title)
    title = re.sub(r'\s+', ' ', title)
    return title.strip()


def get_target_folder(info: dict[str, Any], folder_path: str) -> str | None:
    """Determine the target folder for a video."""
    sanitized_title = sanitize_title(info.get('title', 'Unknown'))
    sanitized_uploader = sanitize_title(info.get('uploader', 'Unknown'))
    upload_date = info.get('upload_date', 'Unknown')
    if upload_date == 'Unknown':
        return None

    output_folder = os.path.join(
        folder_path, sanitized_uploader, f'{upload_date} {sanitized_title}'
    )
    return output_folder


def download_single_video(info: dict[str, Any], folder_path: str, resolution: str = '1080p') -> str | None:
    """Download a single video based on its info dict."""
    check_cancelled()
    output_folder = get_target_folder(info, folder_path)
    if output_folder is None:
        return None
    
    download_mp4 = os.path.join(output_folder, 'download.mp4')
    if os.path.exists(download_mp4):
        try:
            if os.path.getsize(download_mp4) >= 1024:
                logger.info(f'Video already downloaded in {output_folder}')
                return output_folder
            logger.warning(f"Existing download.mp4 looks invalid (too small), will re-download: {download_mp4}")
            os.remove(download_mp4)
        except Exception:
            # Best-effort: proceed to re-download
            pass
    
    sanitized_title = sanitize_title(info.get('title', 'Unknown'))
    sanitized_uploader = sanitize_title(info.get('uploader', 'Unknown'))
    upload_date = info.get('upload_date', 'Unknown')

    resolution_val = resolution.replace('p', '')
    ydl_opts = {
        'format': f'bestvideo[ext=mp4][height<={resolution_val}]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'writeinfojson': True,
        'writethumbnail': True,
        'outtmpl': os.path.join(folder_path, sanitized_uploader, f'{upload_date} {sanitized_title}', 'download'),
        'ignoreerrors': True
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        check_cancelled()
        rc = ydl.download([info['webpage_url']])
        if rc not in (0, None):
            logger.warning(f"yt-dlp returned non-zero code: {rc} ({info.get('webpage_url')})")
    
    if os.path.exists(download_mp4):
        logger.info(f'Video downloaded in {output_folder}')
        return output_folder
    logger.error(f"Download failed, missing file: {download_mp4}")
    return None


def get_info_list_from_url(url: str | list[str], num_videos: int) -> Generator[dict[str, Any], None, None]:
    """Yield video info dictionaries from one or more URLs."""
    if isinstance(url, str):
        url = [url]

    ydl_opts = {
        'format': 'best',
        'dumpjson': True,
        'playlistend': num_videos,
        'ignoreerrors': True
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for u in url:
            check_cancelled()
            result = ydl.extract_info(u, download=False)
            if not result:
                continue
                
            if 'entries' in result:
                for video_info in result['entries']:
                    if video_info:
                        yield video_info
            else:
                yield result


def download_from_url(url: str | list[str], folder_path: str, resolution: str = '1080p', num_videos: int = 5) -> None:
    """Download multiple videos from URLs (wrapper for standalone usage)."""
    if isinstance(url, str):
        url = [url]
    
    for info in get_info_list_from_url(url, num_videos):
        check_cancelled()
        download_single_video(info, folder_path, resolution)
