"""Shared setup and helpers for the experiment scripts. Run scripts from the final/ folder.

input_path(name)     -> audio_in/dry_<name>.wav
save_sound(name, y)  -> writes audio_out/<name>.wav and prints a line
save_fig(fig, name)  -> writes figures/<name>.png and prints a line
"""

from pathlib import Path
import sys

import matplotlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from audio_io import FS, save_audio

# Figure settings for every experiment: save to files only (no pop-up windows),
# keep labels from overlapping, and save at 150 dpi.
matplotlib.use("Agg")
matplotlib.rcParams["figure.constrained_layout.use"] = True
matplotlib.rcParams["savefig.dpi"] = 150

NAMES = ("piano", "voice", "drums")


def input_path(name):
    """audio_in/dry_<name>.wav"""
    path = ROOT / "audio_in" / f"dry_{name}.wav"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: record/import it")
    return path


def save_sound(name, y):
    """Save y as audio_out/<name>.wav and say how long it is."""
    save_audio(ROOT / "audio_out" / f"{name}.wav", y)
    print(f"saved audio_out/{name}.wav  ({len(y) / FS:.2f} s)")


def save_fig(fig, name):
    """Save fig as figures/<name>.png."""
    fig.savefig(ROOT / "figures" / f"{name}.png")
    print(f"saved figures/{name}.png")
