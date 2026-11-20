"""STFT and inverse STFT for the reverb project.

    hann(N)                    -> periodic Hann window, length N
    stft(x, N, hop)            -> S, complex, shape (N//2 + 1, n_frames)
    istft(S, N, hop, length)   -> y, real, length `length`
    frame_times(n_frames, hop) -> center time of each frame in seconds
    bin_freqs(N)               -> frequency of each row of S in Hz

Uses rfft/irfft: real audio has a conjugate-symmetric spectrum (HW3), so
storing bins 0..N/2 is enough, and later edits to S cannot break symmetry.
Frame m is centered on input sample m*hop (N/2 zeros padded in front, HW4).
With a periodic Hann and hop = N/2 the windows sum to 1 (COLA), so istft
is plain overlap-add: no synthesis window, no normalization.
"""

import numpy as np

FS = 44100  # sample rate, Hz


def hann(N):
    """Periodic Hann: 0.5 - 0.5*cos(2*pi*n/N). Divides by N, not N-1."""
    n = np.arange(N)
    return 0.5 - 0.5 * np.cos(2 * np.pi * n / N)


def stft(x, N=2048, hop=1024):
    """Windowed rfft of overlapping frames. One column of S per frame."""
    x = np.asarray(x, dtype=np.float64)
    w = hann(N)
    n_frames = int(np.ceil(len(x) / hop)) + 1
    padded = np.zeros((n_frames - 1) * hop + N)
    padded[N // 2 : N // 2 + len(x)] = x

    S = np.zeros((N // 2 + 1, n_frames), dtype=complex)
    for m in range(n_frames):
        start = m * hop
        S[:, m] = np.fft.rfft(padded[start : start + N] * w)
    return S


def istft(S, N=2048, hop=1024, length=None):
    """irfft each column, overlap-add, drop the front padding, trim."""
    n_frames = S.shape[1]
    out = np.zeros((n_frames - 1) * hop + N)
    for m in range(n_frames):
        start = m * hop
        out[start : start + N] += np.fft.irfft(S[:, m], n=N)
    y = out[N // 2 :]
    return y if length is None else y[:length]


def frame_times(n_frames, hop=1024, fs=FS):
    """Center of frame m is input sample m*hop."""
    return np.arange(n_frames) * hop / fs


def bin_freqs(N=2048, fs=FS):
    """Row k of S is frequency k*fs/N."""
    return np.fft.rfftfreq(N, d=1 / fs)
