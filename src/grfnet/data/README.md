# Datasets

## EEGMMIDB

PhysioNet's EEG Motor Movement/Imagery Dataset [1, 2]: 64-channel EEG from 109
subjects, 14 runs each. Annotations in each EDF file mark rest (`T0`) and the
cued task (`T1`, `T2`).

| Task           | Runs                | Classes (`T1`, `T2`)                         |
| -------------- | ------------------- | -------------------------------------------- |
| `imagery_lr`   | 4, 8, 12            | left fist, right fist                        |
| `imagery_ff`   | 6, 10, 14           | both fists, both feet                        |
| `imagery`      | 4, 6, 8, 10, 12, 14 | left fist, right fist, both fists, both feet |
| `execution_lr` | 3, 7, 11            | left fist, right fist                        |
| `execution_ff` | 5, 9, 13            | both fists, both feet                        |
| `execution`    | 3, 5, 7, 9, 11, 13  | left fist, right fist, both fists, both feet |

Runs 1 and 2 are eyes-open and eyes-closed baselines. `include_rest=True` adds
`T0` as a final `rest` class.

### Setup

Download the [dataset](https://physionet.org/content/eegmmidb/1.0.0/) so that
one folder holds `S001` to `S109`. The loader reads only `.edf` files and finds
the folder from `root=` (or `--root`), then `GRFNET_EEGMMIDB`, then
`src/dataset/files`. Data files are ignored by Git.

### Loading

```python
from grfnet.data import load_epochs

data = load_epochs(task="imagery_lr", subjects="1-10", tmin=0.0, tmax=4.0, cache_dir="runs/cache")
data.X         # (n_trials, 64, 640) float32, microvolts
data.y         # class index into data.class_names
data.subjects  # subject of each trial
```

| Argument           | Default        | Meaning                                                                 |
| ------------------ | -------------- | ----------------------------------------------------------------------- |
| `task`             | `"imagery_lr"` | A task above.                                                           |
| `subjects`         | all            | List, number or text such as `"1-10,12"`.                               |
| `tmin`, `tmax`     | `0.0`, `4.0`   | Window in seconds from the cue; `round((tmax - tmin) * sfreq)` samples. |
| `sfreq`            | `160.0`        | Output rate; other rates are resampled.                                 |
| `l_freq`, `h_freq` | `None`         | Band-pass in Hz, applied before resampling.                             |
| `include_rest`     | `False`        | Add a `rest` class.                                                     |
| `channels`         | all 64         | Channels to keep, in order.                                             |
| `cache_dir`        | `None`         | Per-subject `.npz` cache keyed on these options.                        |
| `progress`         | `False`        | `True` shows a progress bar; `None` only in a terminal.                 |

EDF labels such as `Fc5.` become 10-10 names such as `FC5`, accepted by
`GRFNet(channel_names=...)`. Epochs that overrun a recording are dropped.

### Dataset report

```bash
python -m grfnet.data report --task imagery_lr --cache-dir runs/cache
```

Writes to `runs/dataset_eegmmidb_{task}/`: `report.md`, `summary.json`, `recordings.csv`
(rate, duration and annotation counts per file), and figures of the montage,
class balance, amplitude per subject, example trials, power spectra at C3, Cz
and C4, mu and beta ERD/ERS, and mu-band scalp maps. The default window starts
1 s before the cue to give the ERD/ERS baseline. Needs the `eeg` and `viz`
extras.

### Notes

- Subjects 88, 92 and 100 were recorded at 128 Hz and are resampled. Their cues
  last 5.12 s and rest 1.38 s, against about 4.1 s and 4.2 s elsewhere, so they
  have more trials per run.
- At 160 Hz the default GRFNet dilations become `[1, 1, 3, 5]`; the baseline
  config passes `[1, 2, 3, 5]`.

## References

1. Schalk, G., McFarland, D. J., Hinterberger, T., Birbaumer, N. and Wolpaw, J. R. (2004). BCI2000: a general-purpose brain-computer interface (BCI) system. *IEEE Transactions on Biomedical Engineering*, 51(6), 1034-1043.
2. Goldberger, A. L. et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet: components of a new research resource for complex physiologic signals. *Circulation*, 101(23), e215-e220.
