import json
import os
import re
import time
from typing import Any

from loguru import logger
from openai import OpenAI

from ...config import Settings

# TODO: Move these to Settings/Config if possible
settings = Settings()
_MODEL_NAME = settings.model_name
print(f'Using model {_MODEL_NAME}')




def get_necessary_info(info: dict[str, Any]) -> dict[str, Any]:
    return {
        'title': info.get('title'),
        'uploader': info.get('uploader'),
        'description': info.get('description'),
        'upload_date': info.get('upload_date'),
        'categories': info.get('categories'),
        'tags': info.get('tags'),
    }


def ensure_transcript_length(transcript: str, max_length: int = 4000) -> str:
    if len(transcript) <= max_length:
        return transcript
        
    mid = len(transcript) // 2
    length = max_length // 2
    before = transcript[:mid]
    after = transcript[mid:]
    return before[:length] + after[-length:]


def summarize(info: dict[str, Any], transcript: list[dict[str, Any]], target_language: str = '简体中文') -> dict[str, Any] | None:
    client = OpenAI(
        base_url=settings.openai_api_base or 'https://api.openai.com/v1',
        api_key=settings.openai_api_key
    )
    
    transcript_text = ' '.join(line['text'] for line in transcript)
    transcript_text = ensure_transcript_length(transcript_text, max_length=2000)
    info_message = f'Title: "{info["title"]}" Author: "{info["uploader"]}". '
    
    full_description = (
        f'The following is the full content of the video:\n{info_message}\n{transcript_text}\n{info_message}\n'
        'According to the above content, detailedly Summarize the video in JSON format:\n'
        '```json\n{"title": "", "summary": ""}\n```'
    )
    
    retry_message = ''
    success = False
    summary_data = None
    
    for _ in range(5):
        try:
            messages = [
                {
                    'role': 'system',
                    'content': 'You are a expert in the field of this video. Please summarize the video in JSON format.\n```json\n{"title": "the title of the video", "summary", "the summary of the video"}\n```'
                },
                {'role': 'user', 'content': full_description + retry_message},
            ]
            response = client.chat.completions.create(
                model=_MODEL_NAME,
                messages=messages,
                timeout=240,

            )
            content = response.choices[0].message.content.replace('\n', '')
            
            # Sanity check
            if '视频标题' in content:
                raise Exception("包含“视频标题”")
            
            logger.info(content)
            
            match = re.search(r'\{.*?\}', content)
            if not match:
                raise Exception("No JSON found in response")
                
            summary_json = json.loads(match.group(0))
            summary_data = {
                'title': summary_json['title'].replace('title:', '').strip(),
                'summary': summary_json['summary'].replace('summary:', '').strip()
            }
            
            if 'title' in summary_data['title']:
                 raise Exception('Invalid summary')
                 
            success = True
            break
        except Exception as e:
            retry_message += '\nSummarize the video in JSON format:\n```json\n{"title": "", "summary": ""}\n```'
            logger.warning(f'总结失败\n{e}')
            time.sleep(1)
            
    if not success or not summary_data:
        raise Exception('总结失败')

    title = summary_data['title']
    summary_text = summary_data['summary']
    tags = info.get('tags', [])
    
    messages = [
        {
            'role': 'system',
            'content': f'You are a native speaker of {target_language}. Please translate the title and summary into {target_language} in JSON format. ```json\n{{"title": "the {target_language} title of the video", "summary", "the {target_language} summary of the video", "tags": [list of tags in {target_language}]}}\n```.'
        },
        {
            'role': 'user',
            'content': f'The title of the video is "{title}". The summary of the video is "{summary_text}". Tags: {tags}.\nPlease translate the above title and summary and tags into {target_language} in JSON format. ```json\n{{"title": "", "summary", ""， "tags": []}}\n```. Remember to tranlate the title and the summary and tags into {target_language} in JSON.'
        },
    ]
    
    # Translation retry loop
    # Ideally should limit retries here too
    for _ in range(5):
        try:
            response = client.chat.completions.create(
                model=_MODEL_NAME,
                messages=messages,
                timeout=240,

            )
            content = response.choices[0].message.content.replace('\n', '')
            logger.info(content)
            
            match = re.search(r'\{.*?\}', content)
            if not match:
                 raise Exception("No JSON found")
                 
            summary_json = json.loads(match.group(0))
            
            if target_language in summary_json['title'] or target_language in summary_json['summary']:
                raise Exception('Invalid translation')
                
            title = summary_json['title'].strip()
            
            # Remove quotes if present
            for quote in ['"', '“', '‘', "'", '《']:
                if title.startswith(quote):
                     # Simplified quote stripping
                     title = title.strip(quote + '”’》')
            
            result = {
                'title': title,
                'author': info.get('uploader', ''),
                'summary': summary_json['summary'],
                'tags': summary_json.get('tags', []),
                'language': target_language
            }
            return result
        except Exception as e:
            logger.warning(f'总结翻译失败\n{e}')
            time.sleep(1)
            
    return None


def translation_postprocess(result: str) -> str:
    result = re.sub(r'\（[^)]*\）', '', result)
    result = result.replace('...', '，')
    result = re.sub(r'(?<=\d),(?=\d)', '', result)
    result = result.replace('²', '的平方').replace(
        '————', '：').replace('——', '：').replace('°', '度')
    result = result.replace("AI", '人工智能')
    result = result.replace('变压器', "Transformer")
    return result


def valid_translation(text: str, translation: str) -> tuple[bool, str]:
    if (translation.startswith('```') and translation.endswith('```')):
        translation = translation[3:-3]
        return True, translation_postprocess(translation)

    if (translation.startswith('“') and translation.endswith('”')) or (translation.startswith('"') and translation.endswith('"')):
        translation = translation[1:-1]
        return True, translation_postprocess(translation)

    # Heuristics to remove prefixes like "翻译：“..."
    if '翻译' in translation and '：“' in translation and '”' in translation:
        translation = translation.split('：“')[-1].split('”')[0]
        return True, translation_postprocess(translation)

    if '翻译' in translation and '："' in translation and '"' in translation:
        translation = translation.split('："')[-1].split('"')[0]
        return True, translation_postprocess(translation)

    if '翻译' in translation and ':"' in translation and '"' in translation:
        translation = translation.split('："')[-1].split('"')[0]
        return True, translation_postprocess(translation)

    if len(text) <= 10:
        if len(translation) > 15:
            return False, 'Only translate the following sentence and give me the result.'
    elif len(translation) > len(text) * 0.75:
        # Check if translation is suspiciously long compared to source
        # Note: Original logic: len(translation) > len(text)*0.75 means translation is > 75% of text length? 
        # Actually usually Chinese is shorter than English. If translation is > 75% of text, it might be okay?
        # Original logic seems to forbid translation if it is NOT significantly shorter?
        # Wait: "len(translation) > len(text)*0.75" -> if translation is longer than 0.75 * text.
        # This seems aggressive for short texts.
        # But I will keep original logic for parity.
        return False, 'The translation is too long. Only translate the following sentence and give me the result.'

    forbidden = ['翻译', '这句', '\n', '简体中文', '中文', 'translate', 'Translate', 'translation', 'Translation']
    translation = translation.strip()
    for word in forbidden:
        if word in translation:
            return False, f"Don't include `{word}` in the translation. Only translate the following sentence and give me the result."

    return True, translation_postprocess(translation)


def split_text_into_sentences(para: str) -> list[str]:
    para = re.sub(r'([。！？\?])([^，。！？\?”’》])', r"\1\n\2", para)  # 单字符断句符
    para = re.sub(r'(\.{6})([^，。！？\?”’》])', r"\1\n\2", para)  # 英文省略号
    para = re.sub(r'(\…{2})([^，。！？\?”’》])', r"\1\n\2", para)  # 中文省略号
    para = re.sub(r'([。！？\?][”’])([^，。！？\?”’》])', r'\1\n\2', para)
    para = para.rstrip()
    return para.split("\n")


def split_sentences(translation: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output_data = []
    for item in translation:
        start = item['start']
        text = item['text']
        speaker = item['speaker']
        translation_text = item.get('translation', '')
        sentences = split_text_into_sentences(translation_text)
        
        if not translation_text:
             # Handle empty translation
             duration_per_char = 0
        else:
             duration_per_char = (item['end'] - item['start']) / len(translation_text)
             
        sentence_start_idx = 0
        for sentence in sentences:
            sentence_len = len(sentence)
            sentence_end = start + duration_per_char * sentence_len

            output_data.append({
                "start": round(start, 3),
                "end": round(sentence_end, 3),
                "text": text,
                "speaker": speaker,
                "translation": sentence
            })

            start = sentence_end
            sentence_start_idx += sentence_len
            
    return output_data


def _translate_content(summary: dict[str, Any], transcript: list[dict[str, Any]], target_language: str = '简体中文') -> list[str]:
    client = OpenAI(
        base_url=settings.openai_api_base or 'https://api.openai.com/v1',
        api_key=settings.openai_api_key
    )
    
    info = f'This is a video called "{summary["title"]}". {summary["summary"]}.'
    full_translation = []
    
    fixed_message = [
        {'role': 'system', 'content': f'You are a expert in the field of this video.\n{info}\nTranslate the sentence into {target_language}.下面我让你来充当翻译家，你的目标是把任何语言翻译成中文，请翻译时不要带翻译腔，而是要翻译得自然、流畅和地道，使用优美和高雅的表达方式。请将人工智能的“agent”翻译为“智能体”，强化学习中是`Q-Learning`而不是`Queue Learning`。数学公式写成plain text，不要使用latex。确保翻译正确和简洁。注意信达雅。'},
        {'role': 'user', 'content': '使用地道的中文Translate:"Knowledge is power."'},
        {'role': 'assistant', 'content': '翻译：“知识就是力量。”'},
        {'role': 'user', 'content': '使用地道的中文Translate:"To be or not to be, that is the question."'},
        {'role': 'assistant', 'content': '翻译：“生存还是毁灭，这是一个值得考虑的问题。”'},
    ]
    
    history = []
    for line in transcript:
        text = line['text']
        retry_message = 'Only translate the quoted sentence and give me the final translation.'
        translation = ""
        
        for _ in range(30):
            # Keep history short to avoid token limit
            current_history = history[-30:]
            messages = fixed_message + current_history + [
                {'role': 'user', 'content': f'使用地道的中文Translate:"{text}"'}
            ]
            
            try:
                response = client.chat.completions.create(
                    model=_MODEL_NAME,
                    messages=messages,
                    timeout=240,

                )
                translation = response.choices[0].message.content.replace('\n', '')
                logger.info(f'原文：{text}')
                logger.info(f'译文：{translation}')
                
                is_valid, processed = valid_translation(text, translation)
                if not is_valid:
                    retry_message += processed # processed contains error message in this case?
                    # valid_translation returns (False, error_msg)
                    raise Exception('Invalid translation')
                
                translation = processed
                break
            except Exception as e:
                logger.error(f"Translation error: {e}")
                # Simple retry with new client if internal error (parity with original)
                if str(e) == 'Internal Server Error':
                    client = OpenAI(
                        base_url=settings.openai_api_base or 'https://api.openai.com/v1',
                        api_key=settings.openai_api_key
                    )
                time.sleep(1)
        
        full_translation.append(translation)
        
        history.append({'role': 'user', 'content': f'Translate:"{text}"'})
        history.append({'role': 'assistant', 'content': f'翻译：“{translation}”'})
        # Avoid rate limits
        time.sleep(0.1)

    return full_translation


def translate_folder(folder: str, target_language: str = '简体中文') -> bool:
    if os.path.exists(os.path.join(folder, 'translation.json')):
        logger.info(f'Translation already exists in {folder}')
        return True
    
    info_path = os.path.join(folder, 'download.info.json')
    if not os.path.exists(info_path):
        return False
        
    with open(info_path, 'r', encoding='utf-8') as f:
        info = json.load(f)
    info = get_necessary_info(info)
    
    transcript_path = os.path.join(folder, 'transcript.json')
    if not os.path.exists(transcript_path):
        return False
        
    with open(transcript_path, 'r', encoding='utf-8') as f:
        transcript = json.load(f)
    
    summary_path = os.path.join(folder, 'summary.json')
    if os.path.exists(summary_path):
        with open(summary_path, 'r', encoding='utf-8') as f:
            summary = json.load(f)
    else:
        summary = summarize(info, transcript, target_language)
        if summary is None:
            logger.error(f'Failed to summarize {folder}')
            return False
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    translation_path = os.path.join(folder, 'translation.json')
    
    # Perform translation
    translations = _translate_content(summary, transcript, target_language)
    
    for i, line in enumerate(transcript):
        line['translation'] = translations[i] if i < len(translations) else ""
        
    transcript = split_sentences(transcript)
    
    with open(translation_path, 'w', encoding='utf-8') as f:
        json.dump(transcript, f, indent=2, ensure_ascii=False)
        
    return True


def translate_all_transcript_under_folder(folder: str, target_language: str) -> str:
    count = 0
    for root, dirs, files in os.walk(folder):
        if 'transcript.json' in files and 'translation.json' not in files:
            if translate_folder(root, target_language):
                count += 1
    msg = f'Translated all videos under {folder} (processed {count} files)'
    logger.info(msg)
    return msg
