import json
import os

import requests
from bilibili_toolman.bilisession.common.submission import Submission
from bilibili_toolman.bilisession.web import BiliSession
from dotenv import load_dotenv
from loguru import logger

from ..interrupts import check_cancelled, sleep_with_cancel

load_dotenv()

# 代理配置（仅用于 B 站上传，不影响其他模块）
_BILI_PROXY = os.getenv("BILI_PROXY", "").strip()
if _BILI_PROXY:
    logger.info(f"Bilibili upload proxy configured: {_BILI_PROXY}")

# UPOS 上传 CDN（bilibili_toolman 的 preupload 参数 upcdn）
# 说明：不同 upcdn 会返回不同的 bilivideo 上传入口；有些入口在某些网络/代理下会 TLS 握手失败。
_BILI_UPLOAD_CDN = os.getenv("BILI_UPLOAD_CDN", "").strip()


def _iter_upload_cdns() -> list[str]:
    # 默认顺序：优先用户指定，其次常见可用项
    # 目前验证：bda2/bda/tx 常见可用；cos/ali 在部分网络会出现 SSL unexpected EOF
    candidates = [_BILI_UPLOAD_CDN, "bda2", "bda", "tx"]
    seen: set[str] = set()
    out: list[str] = []
    for c in candidates:
        c = (c or "").strip()
        if not c or c in seen:
            continue
        seen.add(c)
        out.append(c)
    return out


def _upload_video_endpoint_with_fallback(session: BiliSession, video_path: str) -> str:
    last_exc: Exception | None = None
    for cdn in _iter_upload_cdns():
        try:
            session.UPLOAD_CDN = cdn
            logger.info(f"Uploading video via UPOS CDN: {cdn}")
            video_endpoint, _biz_id = session.UploadVideo(video_path)
            return video_endpoint
        except (
            requests.exceptions.SSLError,
            requests.exceptions.ProxyError,
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
        ) as e:
            last_exc = e
            logger.warning(f"UploadVideo failed via CDN={cdn} (network/proxy): {e}")
            continue
        except Exception as e:  # noqa: BLE001 - keep behavior simple, bubble up if not obviously endpoint issue
            last_exc = e
            msg = str(e)
            # 最常见的“入口不可用/上传会话无效”表现：NoSuchUpload / unexpected eof
            if "NoSuchUpload" in msg or "UNEXPECTED_EOF_WHILE_READING" in msg or "unexpected eof" in msg.lower():
                logger.warning(f"UploadVideo failed via CDN={cdn}: {msg}")
                continue
            raise
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("UploadVideo failed: no CDN candidates available")


def bili_login() -> BiliSession:
    sessdata = os.getenv('BILI_SESSDATA')
    bili_jct = os.getenv('BILI_BILI_JCT')
    
    if not sessdata or not bili_jct:
        raise Exception('Please set BILI_SESSDATA and BILI_BILI_JCT in .env file.')
        
    try:
        session = BiliSession(f'SESSDATA={sessdata};bili_jct={bili_jct}')
        # 设置代理（仅影响此 session，不影响其他模块）
        if _BILI_PROXY:
            session.proxies = {
                'http': _BILI_PROXY,
                'https': _BILI_PROXY,
            }
            logger.info(f"Bilibili session using proxy: {_BILI_PROXY}")
            # 代理环境下强烈建议关闭并发分块上传，避免代理/网络抖动导致 multipart 会话失效
            session.WORKERS_UPLOAD = 1
        logger.info("Bilibili session created.")
        return session
    except Exception as e:
        logger.error(e)
        raise Exception('Failed to login to Bilibili. Please check credentials.')


def upload_video(folder: str) -> bool:
    check_cancelled()
    submission_result_path = os.path.join(folder, 'bilibili.json')
    if os.path.exists(submission_result_path):
        with open(submission_result_path, 'r', encoding='utf-8') as f:
            submission_result = json.load(f)
        
        if 'results' in submission_result and submission_result['results'][0]['code'] == 0:
            logger.info('Video already uploaded.')
            return True
        
    video_path = os.path.join(folder, 'video.mp4')
    cover_path = os.path.join(folder, 'video.png')
    
    if not os.path.exists(video_path):
        logger.warning(f"Video file not found: {video_path}")
        return False

    summary_path = os.path.join(folder, 'summary.json')
    if not os.path.exists(summary_path):
        logger.warning(f"Summary file not found: {summary_path}")
        return False

    with open(summary_path, 'r', encoding='utf-8') as f:
        summary = json.load(f)
        
    info_path = os.path.join(folder, 'download.info.json')
    if not os.path.exists(info_path):
        logger.warning(f"Info file not found: {info_path}")
        return False
        
    with open(info_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    summary_title = summary.get('title', 'Untitled').replace('视频标题：', '').strip()
    summary_text = summary.get('summary', '').replace(
        '视频摘要：', '').replace('视频简介：', '').strip()
        
    tags = summary.get('tags', [])
    if not isinstance(tags, list):
        tags = []
        
    author = summary.get("author", "Unknown")
    title = f'【中配】{summary_title} - {author}'
    
    title_english = data.get('title', '')
    webpage_url = data.get('webpage_url', '')
    
    description = (
        f'{title_english}\n{summary_text}\n\n'
        '项目地址：https://github.com/liuzhao1225/YouDub-webui\n'
        'YouDub 是一个开创性的开源工具，旨在将 YouTube 和其他平台上的高质量视频翻译和配音成中文版本。'
        '该工具结合了最新的 AI 技术，包括语音识别、大型语言模型翻译，以及 AI 声音克隆技术，'
        '提供与原视频相似的中文配音，为中文用户提供卓越的观看体验。'
    )

    try:
        session = bili_login()
    except Exception as e:
        logger.error(f"Login skipped/failed: {e}")
        return False
    
    video_endpoint: str | None = None
    for retry in range(5):
        check_cancelled()
        try:
            if video_endpoint is None:
                video_endpoint = _upload_video_endpoint_with_fallback(session, video_path)

            submission = Submission(
                title=title,
                desc=description
            )

            submission.videos.append(
                Submission(
                    title=title,
                    video_endpoint=video_endpoint
                )
            )

            if os.path.exists(cover_path):
                submission.cover_url = session.UploadCover(cover_path)

            base_tags = ['YouDub', author, 'AI', 'ChatGPT', '中文配音', '科学', '科普']
            all_tags = base_tags + tags
            
            seen_tags = set()
            final_tags = []
            for t in all_tags:
                if t not in seen_tags and len(t) <= 20:
                    final_tags.append(t)
                    seen_tags.add(t)
                if len(final_tags) >= 12:
                    break
                    
            for tag in final_tags:
                submission.tags.append(tag)
                
            submission.thread = 201  # 科普 201, 科技
            submission.copyright = submission.COPYRIGHT_REUPLOAD
            submission.source = webpage_url
            
            response = session.SubmitSubmission(submission, seperate_parts=False)
            
            if response['results'][0]['code'] != 0:
                logger.error(response)
                raise Exception(f"Bilibili error: {response}")
                
            logger.info(f"Submission successful: {response}")
            
            with open(submission_result_path, 'w', encoding='utf-8') as f:
                json.dump(response, f, ensure_ascii=False, indent=4)
            return True
            
        except Exception:
            # Use full traceback; upstream (bilibili_toolman) can raise confusing KeyError like: KeyError('OK')
            logger.exception(f"Bilibili upload/submit failed (retry {retry+1}/5)")
            sleep_with_cancel(10)
            
    logger.error("Failed to upload after retries.")
    return False


def upload_all_videos_under_folder(folder: str) -> str:
    count = 0
    for root, _, files in os.walk(folder):
        check_cancelled()
        if 'video.mp4' in files:
            if upload_video(root):
                count += 1
    return f'All videos under {folder} processed. Uploaded count: {count}.'
