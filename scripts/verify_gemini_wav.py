
import os
import sys
import numpy as np
import librosa
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from youdub.core.steps.synthesize_speech import gemini_tts
from youdub.config import Settings

def test_gemini_tts_generation():
    print("Testing Gemini TTS generation and file validity...")
    
    settings = Settings()
    if not settings.gemini_api_key:
        print("Skipping test: GEMINI_API_KEY not set.")
        return

    output_dir = "tests_output"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "test_gemini_audio.wav")
    
    # Clean up previous run
    if os.path.exists(output_path):
        os.remove(output_path)

    text = "This is a test verifying that the audio file has a valid header."
    print(f"Generating audio to {output_path}...")
    
    gemini_tts(text, output_path)
    
    if not os.path.exists(output_path):
        print("FAILURE: Output file was not created.")
        sys.exit(1)
        
    print("File created. Verifying with librosa...")
    try:
        y, sr = librosa.load(output_path, sr=None)
        print(f"SUCCESS: Loaded audio. SR={sr}, Duration={len(y)/sr:.2f}s")
    except Exception as e:
        print(f"FAILURE: librosa could not load the file. Error: {e}")
        sys.exit(1)

    # Double check file header bytes
    with open(output_path, "rb") as f:
        header = f.read(4)
        print(f"File header: {header}")
        if header != b'RIFF':
            print("FAILURE: Header is not RIFF (not a standard WAV).")
            sys.exit(1)
            
    print("ALL TESTS PASSED.")

if __name__ == "__main__":
    test_gemini_tts_generation()
