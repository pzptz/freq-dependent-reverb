"""Measure the RT60 of each room in each band and compare it with the target.

Run:  python experiments/exp07_rt60.py
Writes figures/fig07_rt60.csv and figures/fig06_edc_v2_church.png.
"""

from common import ROOT, save_fig
import csv
import matplotlib.pyplot as plt
import numpy as np

from audio_io import FS
from presets import PRESETS
from reverb_v1 import measure_ir
from reverb_v2 import ir_length, synth_ir
from measure import BANDS, band_filter, band_rt60, edc_db

HEADER = ["band", "church target", "v2 church", "error %", "studio target", "v2 studio",
          "error %", "v1 church", "error %"]


def main():
    rows = table_rows(measure_rooms())
    print_table(rows)
    plot_church_edc()   # fig06
    save_csv(rows)      # fig07


def build_room(preset):
    """The v2 impulse response exactly as reverb_v2 builds it (rt_scale = 1, default seed)."""
    return synth_ir(preset, ir_length(preset))


def measure_rooms():
    """{room: {band: RT60 in s}} for v2 church, v2 studio and v1 church."""
    return {"v2 church": band_rt60(build_room("church")),
            "v2 studio": band_rt60(build_room("studio")),
            "v1 church": band_rt60(measure_ir("church"))}


def error_pct(measured, target):
    """How far measured is from target, in percent. Negative = decays too fast."""
    return (measured - target) / target * 100


def table_rows(rooms):
    """One row per band, in HEADER order: (target, measured, error) for each room.
    v1 is a church too, so it is compared with the church target."""
    rows = []
    for i, band in enumerate(BANDS):
        c_target = PRESETS["church"][i]
        s_target = PRESETS["studio"][i]
        c, s, v1 = rooms["v2 church"][band], rooms["v2 studio"][band], rooms["v1 church"][band]
        rows.append([band,
                     c_target, c, error_pct(c, c_target),
                     s_target, s, error_pct(s, s_target),
                     v1, error_pct(v1, c_target)])
    return rows


def print_table(rows):
    """Seconds with 2 decimals, errors as +1.2%."""
    print(" | ".join(HEADER))
    for band, *values in rows:
        cells = [f"{v:+.1f}%" if column == "error %" else f"{v:.2f}"
                 for column, v in zip(HEADER[1:], values)]
        print(" | ".join([band] + cells))


def plot_church_edc():
    """Each band's energy decay curve (solid) next to its target line (dashed), same color."""
    church = build_room("church")
    t = np.arange(len(church)) / FS
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for k, ((name, (lo, hi)), rt) in enumerate(zip(BANDS.items(), PRESETS["church"])):
        ax.plot(t, edc_db(band_filter(church, lo, hi, FS)), color=f"C{k}", label=name)
        xs, ys = [0, rt], [0, -60]  # Dashed target line: 0 dB at t = 0, -60 dB at t = rt.
        ax.plot(xs, ys, "--", color=f"C{k}", linewidth=1)
    ax.set(xlabel="Time (s)", ylabel="Energy decay (dB)", ylim=(-80, 5),
           xlim=(0, ir_length("church")),
           title="v2 church: measured decay (solid) vs target RT60 (dashed)")
    ax.legend()
    save_fig(fig, "fig06_edc_v2_church")


def save_csv(rows):
    with open(ROOT / "figures" / "fig07_rt60.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        for band, *values in rows:
            writer.writerow([band] + [round(float(v), 2) for v in values])
    print("saved figures/fig07_rt60.csv")


if __name__ == "__main__":
    main()
