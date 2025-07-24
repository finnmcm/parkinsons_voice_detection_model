import numpy as np
from scipy.io import wavfile
from scipy.signal import stft, medfilt, find_peaks
from scipy.ndimage import gaussian_filter1d

def compute_file_threshold(feature_vals: np.ndarray,
                           bins: int = 100,
                           W: float = 5.0) -> float:
    # 1) Build raw histogram
    counts, bin_edges = np.histogram(feature_vals, bins=bins)
    # 2) (Optional) lightly smooth so noise doesn't create tiny bumps
    from scipy.ndimage import gaussian_filter1d
    smooth_counts = gaussian_filter1d(counts.astype(float), sigma=1)

    # 3) Find all interior peaks
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(smooth_counts)

    # 4) Include the 0th bin if it's a local max
    if counts[0] > counts[1]:
        peaks = np.insert(peaks, 0, 0)
    #  (Similarly, you could include the last bin if you ever want it)

    # 5) Need at least two peaks
    if len(peaks) < 2:
        # fallback to mean if histogram is flat or too few peaks
        return float(np.mean(feature_vals))

    # 6) Sort the peak indices by their bin position
    peaks = np.sort(peaks)

    # 7) Convert bin indices to bin centers (feature values)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    M1 = bin_centers[peaks[0]]   # first (silence) peak
    M2 = bin_centers[peaks[1]]   # second (speech) peak

    # 8) Weighted threshold between them
    T = (W * M1 + M2) / (W + 1)
    return float(T)


def analyze_file(path: str, frame_duration: float = 0.02, W: float = 5.0):
    # 1. Load audio via scipy
    sr, audio = wavfile.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(float)
    # Normalize if needed
    if audio.max() > 0:
        audio = audio / np.max(np.abs(audio))
    
    print(f"Audio loaded: {path}")
    print(f"  sr={sr}, dtype={audio.dtype}, min={audio.min():.6f}, max={audio.max():.6f}\n")
    
    # 2. Frame and STFT (no overlap)
    frame_len = int(frame_duration * sr)
    f, t, Zxx = stft(audio,
                     fs=sr,
                     window='hamming',
                     nperseg=frame_len,
                     noverlap=0,
                     boundary=None)
    S = np.abs(Zxx)
    
    # 3. Compute features
    energy = np.sum(S**2, axis=0)
    freqs = f  # frequencies from STFT
    mag = S / (np.sum(S, axis=0, keepdims=True) + 1e-8)
    centroid = np.sum(freqs[:, None] * mag, axis=0)
    spread = np.sqrt(np.sum(((freqs[:, None] - centroid[None, :])**2) * mag, axis=0))
    
    print("Feature ranges:")
    print(f"  Energy: length={len(energy)}, min={energy.min():.6f}, max={energy.max():.6f}")
    print(f"  Spread: length={len(spread)}, min={spread.min():.6f}, max={spread.max():.6f}\n")
    
    # 4. Compute thresholds for this file
    Te = compute_file_threshold(energy, bins=100, W=W)
    Ts = compute_file_threshold(spread, bins=100, W=W)
    
    # 5. Median smoothing (two passes)
    energy_s = medfilt(energy, kernel_size=5)
    energy_s = medfilt(energy_s, kernel_size=5)
    spread_s = medfilt(spread, kernel_size=5)
    spread_s = medfilt(spread_s, kernel_size=5)
    
    # 6. Build masks
    mask_e = energy_s > Te
    mask_s = spread_s > Ts
    speech_mask = mask_e & mask_s
    
    print("Mask statistics:")
    print(f"  Total frames: {len(energy_s)}")
    print(f"  Frames > Te: {mask_e.sum()}")
    print(f"  Frames > Ts: {mask_s.sum()}")
    print(f"  Frames > both: {speech_mask.sum()}\n")
    
    # 7. Segment extraction
    raw = []
    start = None
    for i, v in enumerate(speech_mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            raw.append((start, i))
            start = None
    if start is not None:
        raw.append((start, len(speech_mask)))
    
    # 8. Merge gaps < 0.25s
    merged = []
    max_gap = int(0.25 / frame_duration)
    for s, e in raw:
        if not merged:
            merged.append((s, e))
        else:
            ps, pe = merged[-1]
            if s - pe <= max_gap:
                merged[-1] = (ps, e)
            else:
                merged.append((s, e))
    
    # 9. Drop segments < 1s
    min_frames = int(1.0 / frame_duration)
    final = [(s, e) for s, e in merged if (e - s) >= min_frames]
    print("Detected segments (frame index):", final)
    print("Detected segments (time sec):", [(s*frame_duration, e*frame_duration) for s, e in final])

# Replace with your actual file path below:
analyze_file('data/raw/HY2/ID16_2_dialogue.wav')
