# Experiments

`grfnet.experiments` trains and tests models with subject-disjoint
cross-validation. Every run is defined by one TOML file and writes its
configuration, splits, environment, logs, checkpoints, predictions and metrics
to a run folder.

## Training

```bash
pip install -e ".[eeg,eval]"
```

```bash
python -m grfnet.train configs/eegmmidb_imagery_lr.toml
```

| Option             | Meaning                                                               |
| ------------------ | --------------------------------------------------------------------- |
| `--out DIR`        | Run folder. Default: `runs/<name>`.                                   |
| `--set KEY=VALUE`  | Override a config value. Parsed as TOML, else kept as text. Repeatable. |
| `--folds 0,2`      | Run only these folds. Default: all.                                   |
| `--overwrite`      | Replace a run folder that is not empty.                               |

A quick check on ten subjects, two folds and three epochs:

```bash
python -m grfnet.train configs/eegmmidb_imagery_lr.toml --set data.subjects=1-10 --set split.n_folds=2 --set training.epochs=3 --out runs/smoke
```

## Evaluation

```bash
python -m grfnet.evaluate --checkpoint runs/grfnet_eegmmidb_imagery_lr/fold_0/best.pt
```

This reloads the checkpoint, reads the data with the options it was trained
with, and tests it on that fold's test subjects. `--subjects` tests other
subjects, `--root` points to a moved dataset, and `--out` writes
`metrics.json` and `predictions.csv`.

To compare two runs over the subjects they share, with a two-sided Wilcoxon
signed-rank test on per-subject balanced accuracy (needs the `eval` extra):

```bash
python -m grfnet.evaluate --compare runs/run_a runs/run_b
```

## Protocol

- **Splits.** Subjects are shuffled with `seed` and dealt into `n_folds`
  groups. Each group is the test set of one fold. A `val_fraction` share of the
  remaining subjects is held out for validation. No subject appears in two of
  a fold's train, validation and test sets. The folds are written to
  `splits.json`, and `split.scheme = "file"` reuses a saved file.
- **Normalisation.** `normalize = "global"` divides every trial by one number,
  the standard deviation of the fold's training trials. Relative power between
  channels and between trials is kept. The value is stored in the checkpoint.
- **Training.** AdamW with cross-entropy, cosine learning-rate decay, and
  optional mixed precision on CUDA. Frozen kernels are excluded from the
  optimiser.
- **Model selection.** The checkpoint is the epoch with the highest validation
  balanced accuracy, with ties broken by lower validation loss. Without
  validation subjects it is the last epoch. `patience` stops training after that
  many epochs without improvement.
- **Seeds.** Each fold seeds Python, NumPy and PyTorch with `seed + fold`.
  `deterministic = true` also requests deterministic PyTorch kernels.
- **Metrics.** Accuracy, balanced accuracy, Cohen's kappa, macro F1 and the
  confusion matrix, for each fold, pooled over all folds and for each subject.

## Configuration

Keys left out take the defaults below. TOML has no null value: leaving out an
optional key turns that feature off.

```toml
name = "experiment"        # default run folder name
seed = 0

[data]
dataset = "eegmmidb"
# root = "path/to/files"   # else $GRFNET_EEGMMIDB or src/dataset/files
task = "imagery_lr"
# subjects = "1-109"       # default: every subject found
tmin = 0.0
tmax = 4.0
sfreq = 160.0
# l_freq = 4.0             # band-pass edges in Hz
# h_freq = 40.0
include_rest = false
# channels = ["C3", "Cz", "C4"]
# cache_dir = "runs/cache"
normalize = "global"       # or "none"

[split]
scheme = "kfold"           # or "file"
n_folds = 5
val_fraction = 0.1
# folds = [0]              # run a subset of folds
# file = "runs/other/splits.json"

[model]
name = "GRFNet"
use_channel_names = false  # true passes the data's channel names (geometry seeding)

[model.kwargs]             # any GRFNet constructor argument
# dilations = [1, 2, 3, 5]

[training]
epochs = 100
batch_size = 64
lr = 0.001
weight_decay = 0.0001
scheduler = "cosine"       # or "none"
# patience = 20
device = "auto"            # "cuda", "cpu" or "auto"
amp = false
deterministic = false
```

`n_times`, `n_classes` and `n_channels` come from the data. `sfreq` is passed
to the model unless `model.kwargs` sets `dilations`.

## Run folder

```text
runs/<name>/
  config.toml        resolved configuration
  environment.json   package versions, device, Git commit, start time
  splits.json        subjects of every fold
  results.json       pooled, per-fold and per-subject metrics
  per_subject.csv    per-subject accuracy and balanced accuracy
  fold_<k>/
    history.csv      loss, accuracy, learning rate and time per epoch
    best.pt          selected checkpoint, with the data options and fold subjects
    metrics.json     test metrics
    predictions.csv  subject, run, true class, predicted class and probabilities
```

Run folders live under `runs/`, which Git ignores.
