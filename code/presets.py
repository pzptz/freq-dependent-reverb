"""Room presets: how long each frequency band takes to decay by 60 dB.

    PRESETS["church"] -> [4.5, 3.8, 2.8, 1.8]   RT60 in seconds, one per band
    rt60("church", 8000) -> 1.8   RT60 (s) of the band that 8000 Hz falls in
"""
import numpy as np

BAND_EDGES = [250, 1000, 4000]

PRESETS = {
    #          0~249   250~999 1k~3999  4k+    (Hz)
    "studio": [1.0,    0.9,    0.8,   0.6],
    "church": [4.5,    3.8,    2.8,   1.8],
}


def rt60(preset, freq, rt_scale=1.0):
    values = np.array(PRESETS[preset]) * rt_scale
    band = np.searchsorted(BAND_EDGES, freq, side="right")   # 0, 1, 2 or 3
    return values[band]
