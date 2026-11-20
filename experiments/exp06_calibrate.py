"""Check that the RT60 measurement reads a known answer: 2.0 s in every band.

Run:  python experiments/exp06_calibrate.py
Writes figures/fig05_edc_calib.png.
"""

from common import save_fig
import matplotlib.pyplot as plt
import numpy as np

from audio_io import FS
from measure import BANDS, band_filter, band_rt60, calibration_signal, edc_db

TRUE_RT60 = 2.0


def main():
    x = calibration_signal(TRUE_RT60, 3.0)  # noise that decays with RT60 = 2.0 s in every band
    check_ruler(x)   # does band_rt60 measure 2.0 s?
    plot_edc(x)      # fig05


def check_ruler(x):
    """Measure the RT60 of each band and compare with the true value."""
    print(f"Calibration signal: every band has RT60 = {TRUE_RT60} s")
    for name, rt in band_rt60(x).items():
        err = 100 * (rt - TRUE_RT60) / TRUE_RT60
        print(f"  {name:14s}: measured {rt:.2f} s  (error {err:+.1f}%)")


def plot_edc(x):
    """Energy decay curve of each band, next to the true -60 dB in 2.0 s line."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    t = np.arange(len(x)) / FS
    for name, (lo, hi) in BANDS.items():
        ax.plot(t, edc_db(band_filter(x, lo, hi, FS)), label=name, linewidth=1)
    ax.plot([0, TRUE_RT60], [0, -60], "k--", label=f"RT60 = {TRUE_RT60} s (true)")
    ax.set(xlabel="Time (s)", ylabel="Energy decay (dB)", ylim=(-80, 5), xlim=(0, 3))
    ax.legend()
    save_fig(fig, "fig05_edc_calib")


if __name__ == "__main__":
    main()
