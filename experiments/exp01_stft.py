"""Check the STFT foundation in two steps.

1. COLA: windows at hop 1024 sum to 1; at hop 1536 they do not (fig01).
2. Round trip: istft(stft(voice)) matches the input to < 1e-10.
Writes figures/fig01_cola.png and audio_out/snd01_roundtrip.wav.
"""

from common import input_path, save_fig, save_sound
import matplotlib.pyplot as plt
import numpy as np

from audio_io import FS, load_audio
from stft import hann, stft, istft

N, HOP, BAD_HOP = 2048, 1024, 1536


def main():
    x = load_audio(input_path("voice"))
    check_cola()             # 1. fig01
    check_round_trip(x)      # 2. snd01


def window_sum(hop, n_windows=12):
    """Add shifted copies of the window, as overlap-add does."""
    w = hann(N)
    total = np.zeros((n_windows - 1) * hop + N)
    for m in range(n_windows):
        total[m * hop : m * hop + N] += w
    return total


def check_cola():
    """Do overlapping Hann windows add up to a flat 1? Yes at hop 1024, no at 1536."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    for ax, hop in zip(axes, [HOP, BAD_HOP]):
        total = window_sum(hop)
        middle = total[N : -N]  # ignore the ramps at the two ends
        print(f"hop {hop}: window sum between {middle.min():.4f} and {middle.max():.4f}")
        ax.plot(np.arange(len(total)) / FS, total)
        ax.axhline(1, color="black", linestyle="--", linewidth=0.8)
        ax.set(title=f"Sum of Hann windows, hop = {hop}", xlabel="Time (s)", ylim=(0, 1.15))
    save_fig(fig, "fig01_cola")


def check_round_trip(x):
    """stft then istft should give back exactly the input."""
    S = stft(x, N, HOP)
    y = istft(S, N, HOP, length=len(x))
    error = np.max(np.abs(y - x))
    print(f"round trip: S shape {S.shape}, max error {error:.2e}")
    assert error < 1e-10
    save_sound("snd01_roundtrip", y)


if __name__ == "__main__":
    main()
