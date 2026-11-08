# Datasets

`grfnet.data` loads public datasets into epoched NumPy arrays and writes
dataset reports. It holds no data itself.

## EEG Motor Movement/Imagery Dataset (EEGMMIDB)

PhysioNet's EEGMMIDB [1, 2] holds 64-channel EEG from 109 subjects, each with
14 runs: two one-minute baselines and twelve two-minute task runs. Each EDF file
carries its own annotations: `T0` for rest, and `T1` or `T2` for the cued task.

| Runs        | Task                                 | `T1`       | `T2`        |
| ----------- | ------------------------------------ | ---------- | ----------- |
| 1           | Baseline, eyes open                  |            |             |
| 2           | Baseline, eyes closed                |            |             |
| 3, 7, 11    | Execute: open and close left or right fist | left fist  | right fist  |
| 4, 8, 12    | Imagine: open and close left or right fist | left fist  | right fist  |
| 5, 9, 13    | Execute: open and close both fists or both feet | both fists | both feet |
| 6, 10, 14   | Imagine: open and close both fists or both feet | both fists | both feet |

### Getting the data

Download the dataset from PhysioNet into a folder that holds `S001` to `S109`.
The loader reads only the `.edf` files. It looks for the folder in this order:

1. the `root` argument or `--root` option;
2. the `GRFNET_EEGMMIDB` environment variable;
3. `src/dataset/files`, relative to the working directory.

`src/dataset/` and `*.edf` are ignored by Git.

### Tasks

| Task           | Runs                  | Classes                                          |
| -------------- | --------------------- | ------------------------------------------------ |
| `imagery_lr`   | 4, 8, 12              | left fist, right fist                            |
| `imagery_ff`   | 6, 10, 14             | both fists, both feet                            |
| `imagery`      | 4, 6, 8, 10, 12, 14   | left fist, right fist, both fists, both feet     |
| `execution_lr` | 3, 7, 11              | left fist, right fist                            |
| `execution_ff` | 5, 9, 13              | both fists, both feet                            |
| `execution`    | 3, 5, 7, 9, 11, 13    | left fist, right fist, both fists, both feet     |

`include_rest=True` adds the `T0` periods of the same runs as a final `rest`
class.

### Loading epochs

```python
from grfnet.data import load_epochs

data = load_epochs(task="imagery_lr", subjects="1-10", tmin=0.0, tmax=4.0, cache_dir="runs/cache")
data.X        # (n_trials, 64, 640) float32, microvolts
data.y        # (n_trials,) class index into data.class_names
data.subjects # (n_trials,) subject number, for subject-disjoint splits
```

| Argument            | Default        | Meaning                                                                  |
| ------------------- | -------------- | ------------------------------------------------------------------------ |
| `root`              | see above      | Dataset folder.                                                          |
| `task`              | `"imagery_lr"` | One of the tasks above.                                                  |
| `subjects`          | all            | Subject numbers, as a list or text such as `"1-10,12"`.                  |
| `tmin`, `tmax`      | `0.0`, `4.0`   | Epoch window in seconds from the cue. Length is `round((tmax - tmin) * sfreq)`. |
| `sfreq`             | `160.0`        | Output sampling rate. Recordings at other rates are resampled.           |
| `l_freq`, `h_freq`  | `None`         | Band-pass edges in Hz, applied to the continuous run before resampling.  |
| `include_rest`      | `False`        | Add `T0` periods as a `rest` class.                                      |
| `channels`          | all 64         | Channel names to keep, in order.                                         |
| `cache_dir`         | `None`         | Folder for per-subject `.npz` caches, keyed on the loading options.      |

Channel labels are converted from the EDF form (`Fc5.`) to the 10-10 spelling
(`FC5`), which `channel_positions` and the `GRFNet(channel_names=...)` argument
accept directly. Epochs that run past the end of a recording are dropped.

### Dataset report

```bash
pip install -e ".[eeg,viz]"
```

```bash
python -m grfnet.data report --task imagery_lr --cache-dir runs/cache
```

The report goes to `runs/r0_dataset/` by default. It holds:

| File                 | Content                                                                 |
| -------------------- | ----------------------------------------------------------------------- |
| `report.md`          | Summary table and every figure.                                          |
| `summary.json`       | Trial, class, subject and recording counts, window and channel names.   |
| `recordings.csv`     | Sampling rate, duration and annotation counts of every EDF file.        |
| `montage.png`        | Electrode positions.                                                    |
| `class_balance.png`  | Trials per class per subject.                                           |
| `trial_rms.png`      | Median trial amplitude per subject, with resampled subjects marked.     |
| `example_trials.png` | One raw trial per class at C3, Cz and C4.                               |
| `psd.png`            | Mean post-cue power spectrum per class at C3, Cz and C4.                |
| `erd_ers.png`        | Mu and beta power over time relative to the pre-cue baseline.           |
| `topomaps.png`       | Mu power change from baseline over the scalp, per class.                |

The ERD/ERS and topomap figures need a pre-cue baseline. The report's default
window is -1 s to 4 s for that reason. Run `python -m grfnet.data report --help`
for every option.

### Known properties

- Subjects 88, 92 and 100 were recorded at 128 Hz and the rest at 160 Hz. The
  loader resamples them to `sfreq`. In their task runs, cues last 5.12 s and
  rest periods 1.38 s, against about 4.1 s and 4.2 s for other subjects. They
  therefore have more trials per run.
- At 160 Hz the default GRFNet dilations scale to `[1, 1, 3, 5]`, which
  repeats dilation 1. Pass explicit `dilations` to avoid it.

## References

1. Schalk, G., McFarland, D. J., Hinterberger, T., Birbaumer, N. and Wolpaw,
   J. R. (2004). BCI2000: a general-purpose brain-computer interface (BCI)
   system. *IEEE Transactions on Biomedical Engineering*, 51(6), 1034-1043.
2. Goldberger, A. L. et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet:
   components of a new research resource for complex physiologic signals.
   *Circulation*, 101(23), e215-e220.
