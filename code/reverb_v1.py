"""v1 reverb: give every STFT bin a 'memory' that fades frame by frame.

    reverb_v1(x, preset, mix=0.35, rt_scale=1.0) -> y   (longer than x: the tail)
    measure_ir(preset, rt_scale=1.0) -> h               (v1's response to one click)

For each bin k and frame m:   state = S[k, m] + g[k] * state
with g = 10^(-3 * hop_s / RT60): after RT60 seconds the amplitude is 0.001 (-60 dB).
"""

import numpy as np

from audio_io import FS, mix_signals, normalize_peak
from stft import stft, istft, bin_freqs
from presets import rt60

HOP = 1024  # must match the default hop in stft.py
HOP_S = HOP / FS  # one frame step = 0.02322 s


def decay_per_frame(rt60):
    """Amplitude factor per hop so that the level falls 60 dB in rt60 seconds."""
    return 10 ** (-3 * HOP_S / rt60)


def pad_tail(x, rt):
    """x followed by silence, 1.5 x the longest RT60, so the tail has room to ring out."""
    tail = np.zeros(int(1.5 * rt.max() * FS))
    return np.concatenate([x, tail])


def add_memory(S, g):
    """The v1 idea: each bin keeps a memory that fades by g[k] every frame."""
    state = np.zeros(S.shape[0], dtype=complex)  # memory starts at zero
    V = np.zeros_like(S)
    for m in range(S.shape[1]):
        state = S[:, m] + g * state  # this frame + faded memory
        V[:, m] = state
    return V


def reverb_v1(x, preset, mix=0.35, rt_scale=1.0):
    rt = rt60(preset, bin_freqs(), rt_scale)  # one RT60 per bin, shape (1025,)
    dry = pad_tail(x, rt)
    S = stft(dry)  # shape (1025, n_frames)
    V = add_memory(S, decay_per_frame(rt))
    wet = istft(V, length=len(dry))
    wet *= np.sqrt(np.sum(dry**2) / np.sum(wet**2))  # same energy as dry
    return normalize_peak(mix_signals(dry, wet, mix))


def measure_ir(preset, rt_scale=1.0):
    """v1's response to one click at t = 0: the 'fingerprint' of the reverb."""
    impulse = np.zeros(FS)  # 1 s of silence...
    impulse[0] = 1.0        # ...with one click at the start
    return reverb_v1(impulse, preset, mix=1.0, rt_scale=rt_scale)
