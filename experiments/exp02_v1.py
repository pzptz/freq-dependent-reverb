"""v1 on piano and drums, two presets each.

Run:  python experiments/exp02_v1.py
Writes audio_out/snd02-snd05.
"""

from common import input_path, save_sound
from audio_io import load_audio
from reverb_v1 import reverb_v1

JOBS = [
    ("snd02_piano_v1_studio", "piano", "studio"),
    ("snd03_piano_v1_church", "piano", "church"),
    ("snd04_drums_v1_studio", "drums", "studio"),
    ("snd05_drums_v1_church", "drums", "church"),
]


def main():
    for name, sound, preset in JOBS:
        y = reverb_v1(load_audio(input_path(sound)), preset)
        save_sound(name, y)  # snd02-snd05


if __name__ == "__main__":
    main()
