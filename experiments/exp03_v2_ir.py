"""Build the church room and draw it.

Run:  python experiments/exp03_v2_ir.py
Writes figures/fig02_v1_vs_v2_ir.png and fig03_ir_v2_spec.png.
"""

from common import save_fig
import matplotlib.pyplot as plt
import numpy as np

from audio_io import FS
from stft import stft, frame_times, bin_freqs
from reverb_v1 import measure_ir
from reverb_v2 import ir_length, synth_ir
from measure import to_db


def main():
    church = synth_ir("church", ir_length("church"))  # the church room
    plot_v1_vs_v2(church)        # fig02
    plot_spectrogram(church)     # fig03


def plot_spectrogram(church):
    """Its spectrogram, with the loudest cell at 0 dB and color fixed at -80..0 dB."""
    S = stft(church)
    db = to_db(np.abs(S) ** 2)
    db -= db.max()
    fig, ax = plt.subplots(figsize=(10, 4))
    mesh = ax.pcolormesh(frame_times(S.shape[1]), bin_freqs(), db, vmin=-80, vmax=0, shading="auto")
    ax.set(xlabel="Time (s)", ylabel="Frequency (Hz)", ylim=(0, 8000))
    fig.colorbar(mesh, ax=ax, label="Level (dB)")
    save_fig(fig, "fig03_ir_v2_spec")


def plot_v1_vs_v2(church):
    """The first 200 ms of the v1 and v2 church impulse responses, one above the other."""
    v1 = measure_ir("church")
    n = int(0.2 * FS)
    ms = np.arange(n) / FS * 1000
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    a1.plot(ms, v1[:n], linewidth=0.6)
    a1.set(ylabel="v1")
    a2.plot(ms, church[:n] / np.max(np.abs(church)), linewidth=0.4)
    a2.set(xlabel="Time (ms)", ylabel="v2")
    save_fig(fig, "fig02_v1_vs_v2_ir")


if __name__ == "__main__":
    main()
