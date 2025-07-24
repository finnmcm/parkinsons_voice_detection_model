import os
import glob
import numpy as np
from scipy.io import wavfile
from scipy.signal import stft, medfilt, find_peaks
from scipy.ndimage import gaussian_filter1d
from src.preprocessing.audio_utils import load_audio, save_segment





def compute_file_threshold(feature_vals: np.ndarray,
                           bins: int = 100,
                           W: float = 5.0) -> float:
    """
    Determine a separation threshold between silence and speech for a 1D feature array
    by finding the first two local maxima in its histogram (by index), then using
    T = (W*M1 + M2)/(W+1).

    If fewer than 2 peaks are found, fallback to the mean of feature_vals.
    """
    # 1. Histogram and smoothing
    counts, bin_edges = np.histogram(feature_vals, bins=bins)
    smooth_counts = gaussian_filter1d(counts.astype(float), sigma=1)

    # 2. Find peaks in the smoothed histogram
    peaks, _ = find_peaks(smooth_counts)
    # Include bin 0 if it's a local maximum
    if counts[0] > counts[1]:
        peaks = np.insert(peaks, 0, 0)

    # 3. Need at least two peaks
    if len(peaks) < 2:
        return float(np.mean(feature_vals))

    # 4. Sort peak indices by their bin position (low to high)
    peaks = np.sort(peaks)

    # 5. Compute bin centers and select first two
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    M1 = bin_centers[peaks[0]]
    M2 = bin_centers[peaks[1]]

    # 6. Weighted threshold
    T = (W * M1 + M2) / (W + 1)
    return float(T)


def estimate_global_thresholds(file_list: list[str],
                               frame_duration: float = 0.02,
                               W: float = 5.0,
                               bins: int = 100) -> tuple[float, float]:
    """
    Compute global Te and Ts by averaging per-file thresholds over the given files.
    Only the "reading" files should be passed in file_list.
    """
    Te_list = []
    Ts_list = []

    for path in file_list:
        audio, sr = load_audio(path)
        frame_len = int(frame_duration * sr)
        # STFT with no overlap
        f, t, Zxx = stft(audio,
                         fs=sr,
                         window='hamming',
                         nperseg=frame_len,
                         noverlap=0,
                         boundary=None)
        S = np.abs(Zxx)
        # Short-term energy
        energy = np.sum(S**2, axis=0)
        # Spectral spread
        freqs = f
        mag = S / (np.sum(S, axis=0, keepdims=True) + 1e-8)
        centroid = np.sum(freqs[:, None] * mag, axis=0)
        spread = np.sqrt(np.sum(((freqs[:, None] - centroid[None, :])**2) * mag, axis=0))
        # Compute per-file thresholds
        Te_list.append(compute_file_threshold(energy, bins=bins, W=W))
        Ts_list.append(compute_file_threshold(spread, bins=bins, W=W))

    Te_global = float(np.mean(Te_list))
    Ts_global = float(np.mean(Ts_list))
    return Te_global, Ts_global


def voice_activity_detection(audio: np.ndarray,
                              sr: int,
                              Te_global: float,
                              Ts_global: float,
                              frame_duration: float = 0.02,
                              merge_duration: float = 0.25,
                              min_seg_duration: float = 1.0) -> list[tuple[float, float]]:
    """
    Perform VAD on a single audio array using global thresholds.
    Returns a list of (start_sec, end_sec) segments.
    """
    frame_len = int(frame_duration * sr)
    f, t, Zxx = stft(audio,
                     fs=sr,
                     window='hamming',
                     nperseg=frame_len,
                     noverlap=0,
                     boundary=None)
    S = np.abs(Zxx)
    # Features
    energy = np.sum(S**2, axis=0)
    freqs = f
    mag = S / (np.sum(S, axis=0, keepdims=True) + 1e-8)
    centroid = np.sum(freqs[:, None] * mag, axis=0)
    spread = np.sqrt(np.sum(((freqs[:, None] - centroid[None, :])**2) * mag, axis=0))

    # Median smoothing x2
    energy_s = medfilt(energy, kernel_size=5)
    energy_s = medfilt(energy_s, kernel_size=5)
    spread_s = medfilt(spread, kernel_size=5)
    spread_s = medfilt(spread_s, kernel_size=5)

    # Masks
    mask_e = energy_s > Te_global
    mask_s = spread_s > Ts_global
    speech_mask = mask_e & mask_s

    # Raw segments (frame indices)
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

    # Merge gaps < merge_duration
    merged = []
    max_gap = int(merge_duration / frame_duration)
    for s, e in raw:
        if not merged:
            merged.append((s, e))
        else:
            ps, pe = merged[-1]
            if s - pe <= max_gap:
                merged[-1] = (ps, e)
            else:
                merged.append((s, e))

    # Drop short segments < min_seg_duration
    min_frames = int(min_seg_duration / frame_duration)
    final = [(s, e) for s, e in merged if (e - s) >= min_frames]

    # Convert to seconds
    segments = [(s * frame_duration, e * frame_duration) for s, e in final]
    return segments


def main(data_root: str):
    """
    Run VAD on all reading files under data_root, excluding dialogue.
    """
    # Gather all WAVs
    all_files = glob.glob(os.path.join(data_root, "*.wav"))
    # Filter out 'dialogue' files
    reading_files = [f for f in all_files if 'dialogue' not in os.path.basename(f).lower()]

    print(f"Found {len(reading_files)} reading files for VAD and threshold estimation.")
    # Estimate global thresholds
    Te, Ts = estimate_global_thresholds(reading_files)
    print(f"Global thresholds --> Te: {Te:.6f}, Ts: {Ts:.6f}\n")

    # Apply VAD to each reading file
    for path in sorted(reading_files):
        audio, sr = load_audio(path)
        segs = voice_activity_detection(audio, sr, Te, Ts)
        print(f"{os.path.basename(path)}: {len(segs)} segments -> {segs}")


