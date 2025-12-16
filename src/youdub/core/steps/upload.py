import json
import os
import time

from bilibili_toolman.bilisession.common.submission import Submission
from bilibili_toolman.bilisession.web import BiliSession
from dotenv import load_dotenv
from loguru import logger

# Load environment variables
load_dotenv()


def bili_login() -> BiliSession:
    sessdata = os.getenv('BILI_SESSDATA')
    bili_jct = os.getenv('BILI_BILI_JCT')
    
    if not sessdata or not bili_jct:
        raise Exception('Please set BILI_SESSDATA and BILI_BILI_JCT in .env file.')
        
    try:
        session = BiliSession(f'SESSDATA={sessdata};bili_jct={bili_jct}')
        # We assume login is successful if object is created? 
        # BiliSession might not verify immediately.
        logger.info("Bilibili session created.")
        return session
    except Exception as e:
        logger.error(e)
        raise Exception('Failed to login to Bilibili. Please check credentials.')


def upload_video(folder: str) -> bool:
    submission_result_path = os.path.join(folder, 'bilibili.json')
    if os.path.exists(submission_result_path):
        with open(submission_result_path, 'r', encoding='utf-8') as f:
            submission_result = json.load(f)
        
        # Check if previous upload was successful
        # 'results' key structure depends on library version, assuming parity with original code
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

    # Clean up title and summary
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
    
    # Submit the submission
    for retry in range(5):
        try:
            # Upload video and get endpoint
            video_endpoint, _ = session.UploadVideo(video_path)

            # Create a submission object
            submission = Submission(
                title=title,
                desc=description
            )

            # Add video to submission
            submission.videos.append(
                Submission(
                    title=title,
                    video_endpoint=video_endpoint
                )
            )

            # Upload and set cover
            if os.path.exists(cover_path):
                submission.cover_url = session.UploadCover(cover_path)

            # Tags processing
            # Max 12 tags, max 20 chars per tag
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
            
        except Exception as e:
            logger.error(f"Error submitting (retry {retry+1}/5):\n{e}")
            time.sleep(10)
            
    logger.error("Failed to upload after retries.")
    return False


def upload_all_videos_under_folder(folder: str) -> str:
    count = 0
    for root, _, files in os.walk(folder):
        if 'video.mp4' in files:
            if upload_video(root):
                count += 1
    return f'All videos under {folder} processed. Uploaded count: {count}.'
