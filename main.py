# main.py

import pandas as pd
from src.preprocessing.vad import voice_activity_detection, estimate_global_thresholds
from src.preprocessing.audio_utils import load_audio, save_segment
import os
import matplotlib.pyplot as plt
import numpy as np
import librosa
def build_segments(Te_global, Ts_global):
        
        
    # 1) read your metadata
    #    e.g. metadata.csv has columns: "filepath","label"
    df = pd.read_csv("metadata.csv")

    # 2) loop over each file
    for _, row in df.iterrows():
        path = row.iloc[0]
        label = row.iloc[1]      # HC, HY2, HY3, HY4

        # 3) load the raw waveform
        audio, sr = load_audio(path)
        segments = voice_activity_detection(
        audio, sr, Te_global, Ts_global
    )
        print(label)
        print(segments)
        # 4) run VAD to get (start_frame, end_frame) pairs
        '''
        if label == 'HY3' or label == 'HY4':
            segments = voice_activity_detection(
                audio, sr,
                thresh_scale=0.85,      
                merge_duration=0.20  
            )
        else:
            segments = voice_activity_detection(audio, sr)
        '''
        # 5) convert frame indices → sample indices
        frame_len = int(0.02 * sr)   # must match vad.py’s frame_duration
        for i, (start_sec, end_sec) in enumerate(segments):
            s_start = int(start_sec * sr)
            s_end   = int(end_sec   * sr)

            clip = audio[s_start:s_end]

            # 6) optionally save each clip for inspection or downstream
            out_dir = f"data/processed/{label}"
            out_file = f"{out_dir}/{os.path.basename(path).replace('.wav','')}_seg{i}.wav"
            save_segment(clip, sr, out_file)

            # 7) OR, directly pass this `clip` into your feature‐extractor:
            #     features = extract_scattering_features(clip)
            #     X.append(features); y.append(label)
def main():
    
    df = pd.read_csv("metadata.csv")
    files = []
    # 2) loop over each file
    for _, row in df.iterrows():
        path = row.iloc[0]
        label = row.iloc[1]      # HC, HY2, HY3, HY4
        files.append(path)
    Te_global, Ts_global = estimate_global_thresholds(files)
    build_segments(Te_global, Ts_global)
    
    lst = os.listdir("data/processed/HC") 
    number_files = len(lst)
    print("HC:" + str(number_files))
    lst = os.listdir("data/processed/HY2") 
    number_files = len(lst)
    print("HY2:" + str(number_files))
    lst = os.listdir("data/processed/HY3") 
    number_files = len(lst)
    print("HY3:" + str(number_files))
    lst = os.listdir("data/processed/HY4") 
    number_files = len(lst)
    print("HY4:" + str(number_files))
    

    
main()

'''
main()
lst = os.listdir("data/processed/HC") 
number_files = len(lst)
print("HC:" + str(number_files))
lst = os.listdir("data/processed/HY2") 
number_files = len(lst)
print("HY2:" + str(number_files))
lst = os.listdir("data/processed/HY3") 
number_files = len(lst)
print("HY3:" + str(number_files))
lst = os.listdir("data/processed/HY4") 
number_files = len(lst)
print("HY4:" + str(number_files))
'''





'''
def voice_activity_detection_debug(
    audio: np.ndarray,
    sr: int,
    frame_duration: float = 0.02,
    merge_duration: float = 0.25,
    min_segment_duration: float = 1.0,
    weight: float = 5.0
):
    # 1) framing
    frame_len = int(frame_duration * sr)
    hop = frame_len

    # 2) STFT → magnitude
    S = np.abs(librosa.stft(audio,
                            n_fft=frame_len,
                            hop_length=hop,
                            win_length=frame_len,
                            window='hamming'))

    # 3) raw features
    energy_raw = np.sum(S**2, axis=0)
    freqs = np.linspace(0, sr/2, S.shape[0])
    mag = S / (np.sum(S, axis=0, keepdims=True) + 1e-8)
    centroid = np.sum(freqs[:, None] * mag, axis=0)
    spread_raw = np.sqrt(np.sum(((freqs[:, None] - centroid[None, :])**2) * mag, axis=0))

    # 4) thresholds
    def compute_threshold(x):
        hist, bins = np.histogram(x, bins=100)
        peaks = np.argsort(hist)[-2:]
        m1, m2 = (bins[peaks[0]] + bins[peaks[0]+1]) / 2, \
                 (bins[peaks[1]] + bins[peaks[1]+1]) / 2
        return (weight*m1 + m2)/(weight+1)

    Te = compute_threshold(energy_raw)
    Ts = compute_threshold(spread_raw)

    # 5) smoothing
    energy = medfilt(energy_raw, kernel_size=5)
    spread = medfilt(spread_raw, kernel_size=5)

    # 6) masks
    mask_e = energy > Te
    mask_s = spread > Ts
    speech_mask = mask_e & mask_s

    # 7) raw segments (before merging/filtering)
    raw_segs = []
    start = None
    for i, m in enumerate(speech_mask):
        if m and start is None:
            start = i
        elif not m and start is not None:
            raw_segs.append((start, i))
            start = None
    if start is not None:
        raw_segs.append((start, len(speech_mask)))

    # 8) merge
    merged = []
    max_gap = int(merge_duration/frame_duration)
    for seg in raw_segs:
        if not merged:
            merged.append(seg)
        else:
            ps, pe = merged[-1]
            if seg[0] - pe <= max_gap:
                merged[-1] = (ps, seg[1])
            else:
                merged.append(seg)

    # 9) final filter
    min_len = int(min_segment_duration/frame_duration)
    final = [(s,e) for (s,e) in merged if (e-s) >= min_len]

    return {
        'energy_raw': energy_raw,
        'energy_s': energy,
        'spread_raw': spread_raw,
        'spread_s': spread,
        'Te': Te, 'Ts': Ts,
        'mask_e': mask_e, 'mask_s': mask_s,
        'speech_mask': speech_mask,
        'raw_segments': raw_segs,
        'merged_segments': merged,
        'final_segments': final
    }

def debug(path: str):
    audio, sr = load_audio(path, sr=44100)
    # call your VAD internals directly
    dbg = voice_activity_detection_debug(audio, sr)
    print("Te =", dbg['Te'], "max energy_s =", dbg['energy_s'].max())
    print("Ts =", dbg['Ts'], "max spread_s =", dbg['spread_s'].max())
    print("Frames above energy thresh (smoothed):", dbg['mask_e'].sum())
    print("Frames above spread thresh (smoothed):", dbg['mask_s'].sum())
    print("Frames passing both:", dbg['speech_mask'].sum())
    print("Raw segments:", dbg['raw_segments'])
    print("Merged segments:", dbg['merged_segments'])
    print("Final segments:", dbg['final_segments'])
    
#debug("data/raw/HY3/ID32_3_readtext.wav")
'''