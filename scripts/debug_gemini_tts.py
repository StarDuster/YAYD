from youdub.core.steps.synthesize_speech import gemini_tts_api
from youdub.config import Settings
import wave
import os

print("Testing Gemini TTS...")
settings = Settings()

if not settings.gemini_api_key:
    print("WARNING: GEMINI_API_KEY is not set in .env or environment!")
    print("Please set it to run this test.")
    exit(1)

print(f"Using API Key: {settings.gemini_api_key[:5]}...{settings.gemini_api_key[-3:]}")

try:
    audio = gemini_tts_api("你好，世界！这是一个 Gemini TTS 音频生成测试。", voice_name="Kore")
    
    if audio:
        output_file = "test_gemini.wav"
        with wave.open(output_file, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            wf.writeframes(audio)
        print(f"Success! Saved audio to {output_file}")
        print(f"File size: {os.path.getsize(output_file)} bytes")
    else:
        print("Failed! No audio data returned.")

except Exception as e:
    print(f"An error occurred: {e}")
