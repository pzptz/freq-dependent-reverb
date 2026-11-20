"""Measure RT60 the standard acoustics way (Schroeder backward integration).

to_db(energy)         -> energy in dB (safe for zeros)
fit_slope(t, db)      -> slope of the best straight line through (t, db)
calibration_signal(rt60, length_s) -> noise that decays with a known RT60
edc_db(h)                           -> energy decay curve in dB (starts at 0)
rt60_from_edc(edc, fs, lo, hi)      -> RT60 from a straight-line fit (T30)
band_rt60(h, fs)                    -> {band: RT60} for the four bands
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt

from audio_io import FS
from reverb_v2 import make_noise

BANDS = {
    "0-250 Hz": (None, 250),
    "250-1000 Hz": (250, 1000),
    "1000-4000 Hz": (1000, 4000),
    "above 4000 Hz": (4000, None),
}


def to_db(energy):
    """Energy -> dB. Energy, not amplitude: 10 * log10. Never take log of 0."""
    return 10 * np.log10(np.maximum(energy, 1e-12))


def fit_slope(t, db):
    """Slope of the best straight line through the points (t, db)."""
    slope, _ = np.polyfit(t, db, 1)
    return slope


def calibration_signal(rt60, length_s, seed=1):
    """Noise whose amplitude falls 60 dB in rt60 seconds, at every frequency."""
    t = np.arange(int(length_s * FS)) / FS
    noise = make_noise(length_s, seed)
    return noise * 10 ** (-3 * t / rt60)


def edc_db(h):
    """Schroeder's energy decay curve: energy from each sample to the END, in dB."""
    energy = h**2
    edc = np.flip(np.cumsum(np.flip(energy)))
    return to_db(edc / edc[0])


def rt60_from_edc(edc, fs, lo=-5, hi=-35):
    """RT60 from the straight part of the EDC, between lo and hi dB (T30: -5 to -35)."""
    samples = np.where((edc <= lo) & (edc >= hi))[0]
    slope = fit_slope(samples / fs, edc[samples])
    return -60 / slope


def band_filter(h, lo, hi, fs):
    """Keep only lo..hi Hz of h. 4th-order Butterworth, run forward and backward."""
    if lo is None:
        sos = butter(4, hi, btype="lowpass", fs=fs, output="sos")
    elif hi is None:
        sos = butter(4, lo, btype="highpass", fs=fs, output="sos")
    else:
        sos = butter(4, [lo, hi], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, h)  # forward + backward: no time delay


def band_rt60(h, fs=FS):
    """RT60 in each of the four bands, measured independently of the STFT."""
    return {
        name: rt60_from_edc(edc_db(band_filter(h, lo, hi, fs)), fs)
        for name, (lo, hi) in BANDS.items()
    }
