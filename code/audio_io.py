"""Read and write mono WAV files at the project sample rate (44,100 Hz)."""

import warnings
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample

FS = 44100


def mix_signals(dry, wet, mix):
    """Blend: mix = 0 -> all dry, mix = 1 -> all wet."""
    return (1 - mix) * dry + mix * wet


def normalize_peak(y, peak=0.9):
    """Scale y so that its largest absolute value equals peak."""
    largest = np.max(np.abs(y))
    if largest == 0:
        return y
    return y * peak / largest


def load_audio(path):
    rate, data = wavfile.read(path)
    x = (data[:, 0] if data.ndim == 2 else data).astype(np.float64)
    if rate != FS:
        x = resample(x, round(len(x) * FS / rate))
    return normalize_peak(x)


def save_audio(path, y):
    y = np.asarray(y, dtype=np.float64).copy()
    peak = np.max(np.abs(y)) if y.size else 0.0
    if peak > 0.99:
        warnings.warn(f"{path}: peak {peak:.3f} scaled down to 0.99")
        y *= 0.99 / peak
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(path, FS, y.astype(np.float32))
    return y
