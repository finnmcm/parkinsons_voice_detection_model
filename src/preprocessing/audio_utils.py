# src/preprocessing/audio_utils.py

import librosa
import soundfile as sf
import os
import numpy as np
from scipy.io import wavfile

def load_audio(path: str) -> tuple[np.ndarray, int]:
    """
    Load a WAV file as mono, normalized to [-1,1].
    Returns (audio, sr).
    """
    sr, audio = wavfile.read(path)
    # If stereo, average to mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    # Convert to float in [-1,1]
    audio = audio.astype(np.float64)
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio /= max_val
    return audio, sr

def save_segment(waveform: np.ndarray, sr: int, out_path: str):
    """
    Write a small waveform clip to disk as WAV.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sf.write(out_path, waveform, sr)
