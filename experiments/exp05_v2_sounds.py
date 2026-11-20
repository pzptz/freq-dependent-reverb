"""Make all the v2 sound examples.

Run:  python experiments/exp05_v2_sounds.py
Writes audio_out/snd06-snd11 (3 instruments x 2 rooms),
snd12-14 (the 3 mix values) and snd15-17 (the 3 reverb-time values).
"""

from common import NAMES, input_path, save_sound
from audio_io import load_audio
from reverb_v2 import reverb_v2

MIX_VALUES = [0.1, 0.55, 0.9]
RT_SCALES = [0.5, 1.0, 2.5]


def main():
    sounds = {name: load_audio(input_path(name)) for name in NAMES}
    rooms(sounds)               # snd06-snd11
    mix_sweep(sounds["voice"])  # snd12-snd14
    rt_sweep(sounds["piano"])   # snd15-snd17


def rooms(sounds):
    """Every instrument in both rooms, default settings."""
    number = 6
    for name in NAMES:
        for preset in ["studio", "church"]:
            save_sound(f"snd{number:02d}_{name}_v2_{preset}", reverb_v2(sounds[name], preset))
            number += 1


def mix_sweep(voice):
    """Voice in the church, from mostly dry to mostly wet."""
    for number, mix in zip([12, 13, 14], MIX_VALUES):
        save_sound(f"snd{number}_voice_church_mix{mix}", reverb_v2(voice, "church", mix=mix))


def rt_sweep(piano):
    """Piano in a shorter, normal and longer-ringing church."""
    for number, scale in zip([15, 16, 17], RT_SCALES):
        save_sound(f"snd{number}_piano_church_rt{scale}", reverb_v2(piano, "church", rt_scale=scale))


if __name__ == "__main__":
    main()
