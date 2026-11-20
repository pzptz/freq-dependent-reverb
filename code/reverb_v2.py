"""v2 reverb: build a room's impulse response from decaying noise, then
convolve the dry sound with it.

make_noise(length_s, seed)           -> white noise, length_s seconds
decay_matrix(t, rt)                  -> A, shape (len(rt), len(t))
synth_ir(preset, length_s, rt_scale) -> impulse response, energy 1
ir_length(preset, rt_scale)          -> impulse response length in seconds
reverb_v2(x, preset, mix, rt_scale)  -> the reverberated sound (longer than x)

Idea (Moorer 1979, p. 26): a good hall's impulse response sounds like white
noise with an exponentially decaying envelope. Here every STFT bin gets its
own decay, so highs can fade faster than lows.
"""

import numpy as np
from scipy.signal import fftconvolve

from audio_io import FS, mix_signals, normalize_peak
from presets import PRESETS, rt60
from stft import stft, istft, frame_times, bin_freqs


def make_noise(length_s, seed=0):
    """White noise, length_s seconds long. Same seed -> same noise."""
    n = int(length_s * FS)
    rng = np.random.default_rng(seed)
    return rng.standard_normal(n)


def decay_matrix(t, rt):
    """A[k, m] = 10^(-3 * t[m] / rt[k]): one row per frequency, one column per frame."""
    return 10 ** (-3 * t.reshape(1, -1) / rt.reshape(-1, 1))


def synth_ir(preset, length_s, rt_scale=1.0, seed=0):
    """Decaying noise whose decay time depends on frequency. Scaled to energy 1."""
    noise = make_noise(length_s, seed)  # Step 1: white noise
    S = stft(noise)  # Step 2: noise -> grid (bins x frames)
    t = frame_times(S.shape[1])  # Step 3: time of each column (s)
    rt = rt60(preset, bin_freqs(), rt_scale)  #         RT60 of each row (s)
    A = decay_matrix(t, rt)  # Step 4: fade factor for every cell
    h = istft(S * A, length=len(noise))  # Step 5: fade the grid, back to sound
    return h / np.sqrt(np.sum(h**2))  # Step 6: scale to total energy 1


def ir_length(preset, rt_scale=1.0):
    """How long the room's impulse response should be: 1.5 x its longest RT60, x rt_scale."""
    longest = max(PRESETS[preset])
    return 1.5 * longest * rt_scale


def reverb_v2(x, preset, mix=0.35, rt_scale=1.0):
    ir = synth_ir(preset, ir_length(preset, rt_scale), rt_scale)  # the room
    wet = fftconvolve(x, ir)  # dry sound in the room
    dry = np.concatenate([x, np.zeros(len(wet) - len(x))])  # same length as wet
    return normalize_peak(mix_signals(dry, wet, mix))  # blend, then set the peak
