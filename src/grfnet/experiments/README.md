# Experiments

Training, evaluation and reports for subject-disjoint cross-validation. Each run
is defined by one TOML config and writes everything it produces to
`runs/<run name>/`.

## Usage

```bash
python -m grfnet.train configs/grfnet_eegmmidb_imagery_lr.toml                # train and test all folds
python -m grfnet.train configs/grfnet_eegmmidb_imagery_lr.toml --set training.epochs=3 --folds 0 --set variant=smoke
python -m grfnet.evaluate --checkpoint runs/grfnet_eegmmidb_imagery_lr/fold_0/best.pt
python -m grfnet.evaluate --compare runs/grfnet_eegmmidb_imagery_lr runs/grfnet_eegmmidb_imagery_lr_geometry
python -m grfnet.report runs/grfnet_eegmmidb_imagery_lr                       # figures, tables, benchmark
```

| Command    | Options                                                                                                                                  |
| ---------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `train`    | `--out DIR`, `--set KEY=VALUE` (TOML value, else text; repeatable), `--folds 0,2`, `--overwrite`, `--no-progress`                          |
| `evaluate` | `--checkpoint PATH` or `--compare RUN_A RUN_B`; `--subjects`, `--root`, `--out`, `--metric`, `--device`                                   |
| `report`   | `--out DIR`, `--fold K`, `--root`, `--no-data`, `--no-benchmark`, `--max-trials N`, `--dpi`, `--device`, `--no-progress`                   |

Training shows a tqdm progress bar per fold in a terminal, and one line per
epoch otherwise. `evaluate --checkpoint` reloads the data with the options the
model was trained with and tests the fold's own test subjects by default.

## Run names

Runs are named `{model}_{dataset}_{task}` in lowercase, with `_rest` when the
rest class is included and `_{variant}` for a variant of a config:

| Output         | Location                                          | Example                                        |
| -------------- | ------------------------------------------------- | ---------------------------------------------- |
| Config         | `configs/{model}_{dataset}_{task}.toml`           | `configs/grfnet_eegmmidb_imagery_lr.toml`      |
| Training run   | `runs/{model}_{dataset}_{task}[_{variant}]/`      | `runs/grfnet_eegmmidb_imagery_lr_geometry/`    |
| Run report     | `runs/{run}/report/`                              | `runs/grfnet_eegmmidb_imagery_lr/report/`      |
| Dataset report | `runs/dataset_{dataset}_{task}/`                  | `runs/dataset_eegmmidb_imagery_lr/`            |
| Epoch cache    | `runs/cache/`                                     |                                                |

Set `variant` in the config or with `--set variant=geometry`. `name` overrides
the whole run name and `--out` the folder; both should be the exception.

## Protocol

- **Splits.** Subjects are shuffled with `seed` and dealt into `n_folds` test
  groups. `val_fraction` of the remaining subjects is held out for validation.
  No subject is in two sets of a fold. Folds are saved to `splits.json`.
- **Normalisation.** Trials are divided by the standard deviation of the fold's
  training trials, keeping relative power between channels.
- **Training.** AdamW, cross-entropy and cosine decay. Frozen kernels are not
  optimised.
- **Selection.** The epoch with the highest validation balanced accuracy,
  ties broken by validation loss, is tested. `patience` stops training early.
- **Seeds.** Python, NumPy and PyTorch are seeded with `seed + fold`.
- **Metrics.** Accuracy, balanced accuracy, Cohen's kappa, macro F1 and the
  confusion matrix, per fold, pooled and per subject.

## Configuration

Omitted keys take these defaults. TOML has no null: leaving out an optional key
turns it off.

```toml
# name = "..."              # default: {model}_{dataset}_{task}[_rest][_{variant}]
# variant = "geometry"
seed = 0

[data]
dataset = "eegmmidb"
task = "imagery_lr"
tmin = 0.0
tmax = 4.0
sfreq = 160.0
include_rest = false
normalize = "global"        # or "none"
# root, subjects, l_freq, h_freq, channels, cache_dir: see the datasets README

[split]
scheme = "kfold"            # or "file", with file = "path/to/splits.json"
n_folds = 5
val_fraction = 0.1
# folds = [0]

[model]
name = "GRFNet"
use_channel_names = false   # true seeds spatial filters from the data's channel names

[model.kwargs]              # any GRFNet argument, e.g. dilations = [1, 2, 3, 5]

[training]
epochs = 100
batch_size = 64
lr = 0.001
weight_decay = 0.0001
scheduler = "cosine"        # or "none"
# patience = 20
device = "auto"
amp = false
deterministic = false
```

`n_channels`, `n_times` and `n_classes` come from the data; `sfreq` is passed to
the model unless `dilations` is set.

## Outputs

```text
runs/<run name>/
  config.toml, environment.json, splits.json
  results.json, per_subject.csv        pooled, per-fold and per-subject metrics
  fold_<k>/
    history.csv                        per-epoch loss, accuracy, learning rate, time
    best.pt                            selected checkpoint with data options and subjects
    metrics.json, predictions.csv
  report/                              python -m grfnet.report
```

| Report file            | Content                                                                                       |
| ---------------------- | --------------------------------------------------------------------------------------------- |
| `report.md`, `.json`   | Results, per-fold table, efficiency and all figures.                                          |
| `efficiency.json`      | Parameters, FLOPs per trial, checkpoint size, CPU and CUDA latency and throughput, training time, machine. |
| `learning_curves.png`  | Loss and accuracy per fold; dots mark the selected epoch.                                     |
| `confusion_matrix.png` | Pooled test confusion matrix.                                                                 |
| `subject_scores.png`   | Balanced accuracy per subject and per fold, against chance.                                   |
| `stream_gates.png`     | Gate weight of each pooling statistic per fold.                                               |
| `spatial_filters.png`  | Spatial filter weights.                                                                       |
| `kernel_responses.png` | Frequency responses of frozen and learnable kernels per dilation.                             |
| `spatial_patterns.png` | Activation patterns `A = C W^T (W C W^T)^-1` from the test data [1]. *                        |
| `embeddings.png`       | t-SNE of the embedding (PCA without scikit-learn), by class and by correctness. *             |
| `saliency.png`         | Mean \|gradient x input\| per channel and over time. *                                        |

Model figures use the lowest fold's checkpoint unless `--fold` is given. Figures
marked * read that fold's test data and are skipped, with a note, if the dataset
is missing. Timings depend on the machine.

## References

1. Haufe, S. et al. (2014). On the interpretation of weight vectors of linear models in multivariate neuroimaging. *NeuroImage*, 87, 96-110.
