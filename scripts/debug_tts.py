from youdub.core.steps.synthesize_speech import bytedance_tts_api
from youdub.config import Settings
import logging

# Configure logging to see the warning
logging.basicConfig(level=logging.WARNING)

print("Testing ByteDance TTS API...")
settings = Settings()
print(f"AppID: {settings.bytedance_appid}")
# Obscure token
token = settings.bytedance_access_token
print(f"Token: {token[:5]}...{token[-5:] if token else ''}")

text = "测试语音合成。"
try:
    result = bytedance_tts_api(text, voice_type='BV701_streaming')
    if result:
        print("Success! Audio data received.")
    else:
        print("Failed: No audio data returned.")
except Exception as e:
    print(f"Exception: {e}")
