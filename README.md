# Code-based frequency-dependent Reverb using the DFT

Peter Li

An offline Python tool that takes a WAV file and returns a
reverberated one, with a different decay time for each frequency band. It
builds on HW3 (FFT), HW4 (spectrogram / STFT) and HW5 (damped comb filter).

Two methods are included:

- **v1:** a per-bin recursion across STFT frames that builds on HW5's comb filter.
- **v2 (main method):** white noise goes through the STFT, every cell is faded
  with a frequency-dependent decay, and the inverse STFT gives the room's
  impulse response. The dry sound is then convolved with that response using
  `scipy.signal.fftconvolve`.

Rooms: `studio` and `church` (RT60 per band in `code/presets.py`). Knobs:
`mix` (wet/dry) and `rt_scale` (multiplies every RT60 of the preset).

## Setup (once)

The project uses a conda environment named `music320` (Python 3.13; developed
on 3.13.2).

```bash
cd final
conda create -n music320 python=3.13
conda activate music320
python -m pip install -r requirements.txt
```

Use `python -m pip` so packages go into the active environment. The
environment lives outside `final/`, so there is nothing to exclude when you zip
the project. Run every command below from the `final/` folder.

## Audio recordings

Put three WAV files directly in `audio_in/`, named exactly:

```text
audio_in/dry_piano.wav
audio_in/dry_voice.wav
audio_in/dry_drums.wav
```

Convert phone M4A files to WAV first. Every script reads audio through
`load_audio`, which keeps the first channel, resamples to 44.1 kHz if needed,
and scales the peak to 0.9, so the files do not need any other preparation.

## Run (from the final/ folder)

```bash
python experiments/exp01_stft.py             # COLA, round trip (fig01, snd01)
python experiments/exp02_v1.py               # v1 on piano and drums (snd02-05)
python experiments/exp03_v2_ir.py            # v2 church room: v1-vs-v2 click response and spectrogram (fig02, fig03)
python experiments/exp04_spectrograms.py     # piano in four versions: dry, v2 studio, v1 church, v2 church (fig04)
python experiments/exp05_v2_sounds.py        # 12 v2 sounds (snd06-17); mix and rt_scale choices at the top of the file
python experiments/exp06_calibrate.py        # RT60 ruler on a known 2.0 s decay (fig05)
python experiments/exp07_rt60.py             # RT60 per room and band vs target (figures/fig07_rt60.csv, fig06)
```

## Layout

```text
final/
  code/
    audio_io.py     reading and writing mono 44.1 kHz WAV files
    stft.py         STFT and inverse STFT
    presets.py      RT60 per band for the studio and church rooms
    reverb_v1.py    v1: per-bin recursion across frames
    reverb_v2.py    v2: impulse response from decaying noise, then convolution
    measure.py      RT60 ruler: measured decay time per band
  experiments/      exp01-exp07 as above, common.py (paths, figure settings, save helpers)
  audio_in/         input dry audio recordings
  audio_out/        sndNN_*.wav outputs
  figures/          figNN_*.png outputs, fig07_rt60.csv (RT60 table)
```

## Conventions

- 44.1 kHz, mono, N = 2048, hop = 1024 (bin spacing 21.5 Hz, 1025 `rfft` rows).
- Periodic Hann window; with hop = N/2 the windows add to exactly 1 (COLA), so
  `istft(stft(x))` returns `x` to about 1e-15.
- RT60 = time to fall 60 dB. Amplitude fade is `10^(-3 t / RT60)`.
- Band edges for the presets: 250, 1000 and 4000 Hz.
