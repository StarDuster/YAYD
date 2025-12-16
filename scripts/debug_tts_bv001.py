from youdub.core.steps.synthesize_speech import bytedance_tts_api
import logging

logging.basicConfig(level=logging.WARNING)

print("Testing ByteDance TTS API with BV001_streaming...")
try:
    result = bytedance_tts_api("测试语音合成。", voice_type='BV001_streaming')
    if result:
        print("BV001_streaming: Success!")
    else:
        print("BV001_streaming: Failed.")
except Exception as e:
    print(f"Exception: {e}")
