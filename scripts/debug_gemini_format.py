
import os
import sys
from youdub.config import Settings
from youdub.core.steps.synthesize_speech import _get_gemini_client
from google import genai
from google.genai import types

def debug_gemini_tts():
    settings = Settings()
    client = _get_gemini_client()
    if not client:
        print("Failed to initialize Gemini client")
        return

    text = "Hello, this is a test."
    voice_name = settings.gemini_tts_voice or "Kore"
    model_name = settings.gemini_tts_model or "gemini-2.5-flash-preview-tts"

    print(f"Generating audio using model: {model_name}, voice: {voice_name}")
    
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=text,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=voice_name,
                        )
                    )
                ),
            )
        )
        
        if (response.candidates and 
            response.candidates[0].content and 
            response.candidates[0].content.parts and 
            response.candidates[0].content.parts[0].inline_data):
            
            data = response.candidates[0].content.parts[0].inline_data.data
            print(f"Received {len(data)} bytes")
            print(f"First 16 bytes: {data[:16]}")
            
            # Save raw dump
            with open("debug_gemini_output.bin", "wb") as f:
                f.write(data)
            print("Saved debug_gemini_output.bin")
            
            # Check for RIFF header
            if data.startswith(b'RIFF'):
                print("Header detection: Looks like WAV (RIFF)")
            elif data.startswith(b'\xFF\xF3') or data.startswith(b'\xFF\xF2') or data.startswith(b'ID3'):
                print("Header detection: Looks like MP3")
            elif data.startswith(b'OggS'):
                print("Header detection: Looks like OGG")
            else:
                print("Header detection: Unknown format")
                
        else:
            print("No audio data in response")
            print(response)

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    debug_gemini_tts()
