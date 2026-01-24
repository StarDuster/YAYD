import re
import string
import numpy as np
from scipy.io import wavfile

def sanitize_filename(filename: str) -> str:
    valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
    sanitized_filename = "".join(c for c in filename if c in valid_chars)
    sanitized_filename = re.sub(" +", " ", sanitized_filename)
    return sanitized_filename.strip()


def save_wav(wav: np.ndarray, output_path: str, sample_rate: int = 24000) -> None:
    wav_i16 = wav * 32767
    wavfile.write(output_path, sample_rate, wav_i16.astype(np.int16))

def save_wav_norm(wav: np.ndarray, output_path: str, sample_rate: int = 24000) -> None:
    wav_norm = wav * (32767 / max(0.01, np.max(np.abs(wav))))
    wavfile.write(output_path, sample_rate, wav_norm.astype(np.int16))

def normalize_wav(wav_path: str) -> None:
    try:
        sample_rate, wav = wavfile.read(wav_path)
        wav_norm = wav * (32767 / max(0.01, np.max(np.abs(wav))))
        wavfile.write(wav_path, sample_rate, wav_norm.astype(np.int16))
    except Exception as e:
        print(f"Error normalizing wav {wav_path}: {e}")

