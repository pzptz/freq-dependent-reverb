"""Spectrograms of the piano: dry, v2 studio, v1 church and v2 church, on one color scale.

Run:  python experiments/exp04_spectrograms.py
Writes figures/fig04_spectrograms.png.
"""

from common import input_path, save_fig
import matplotlib.pyplot as plt
import numpy as np

from audio_io import FS, load_audio
from stft import stft, frame_times, bin_freqs
from measure import to_db
from reverb_v1 import reverb_v1
from reverb_v2 import reverb_v2

CLIP_S = 6.0  # first 6 s of the piano: long enough to see the tails


def main():
    x = load_audio(input_path("piano"))[: int(CLIP_S * FS)]
    versions = make_versions(x)
    plot_grid(versions)  # fig04


def spectrogram_db(x):
    """Energy of every STFT cell in dB (not yet relative to anything)."""
    return to_db(np.abs(stft(x)) ** 2)


def relative_to_loudest(dbs):
    """Shift every spectrogram by the SAME amount so the loudest cell of all four is 0 dB."""
    loudest = np.max(dbs)
    return dbs - loudest


def pad_to(y, n):
    """y followed by zeros, so that it is n samples long."""
    return np.concatenate([y, np.zeros(n - len(y))])


def make_versions(x):
    """The four versions, in the order they are drawn."""
    return {
        "dry": x,
        "v2 studio": reverb_v2(x, "studio"),
        "v1 church": reverb_v1(x, "church"),
        "v2 church": reverb_v2(x, "church"),
    }


def plot_grid(versions):
    """2 x 2 spectrograms on one time axis, one frequency axis and one color scale."""
    n = max(len(y) for y in versions.values())  # the longest version (v1/v2 church)
    padded = {name: pad_to(y, n) for name, y in versions.items()}
    dbs = relative_to_loudest([spectrogram_db(y) for y in padded.values()])
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True, sharey=True)
    for ax, name, db in zip(axes.flat, padded, dbs):
        mesh = ax.pcolormesh(
            frame_times(db.shape[1]), bin_freqs(), db, vmin=-80, vmax=0, shading="auto"
        )
        ax.set(title=name, xlim=(0, n / FS), ylim=(0, 8000))
    for ax in axes[1]:
        ax.set_xlabel("Time (s)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Frequency (Hz)")
    fig.colorbar(mesh, ax=axes, label="Level (dB, 0 = loudest cell of all four)")
    print(f"all four padded to {n / FS:.2f} s")
    save_fig(fig, "fig04_spectrograms")


if __name__ == "__main__":
    main()
