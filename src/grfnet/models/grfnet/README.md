# GRFNet

GRFNet maps one multichannel sequence to one output vector: class scores,
regression targets, multi-label scores or an embedding. It is built for small
datasets, where a large learned feature extractor would overfit, and for
channels with a physical layout, such as EEG electrodes.

## Architecture

| Stage             | Operation                                                                                         |
| ----------------- | ------------------------------------------------------------------------------------------------- |
| Spatial front-end | 1x1 convolution and BatchNorm mixing the channels into `spatial_components` components.            |
| Kernel bank       | Depthwise dilated convolutions per component: frozen mean-centred random kernels [1] plus learnable ones. |
| Fusion pooling    | Each feature map summarised over time by the chosen statistics.                                   |
| Gated embedding   | A softmax gate weights each statistic. `forward_features` returns this embedding.                 |
| Head              | Dropout and a linear layer by default; replaceable or removable.                                  |

Input `(batch, n_channels, n_times)` of any length; output `(batch, n_outputs)`,
or `(batch, embedding_dim)` with `head=None`. The model is stateless. The
front-end and bank are `SpatialFrontEnd` and `RocketFeatureBank` in
[`layers.py`](../../layers.py).

| Channels | Outputs | Trainable parameters | Frozen weights |
| -------- | ------- | -------------------- | -------------- |
| 3        | 2       | 2,389                | 1,152          |
| 22       | 4       | 3,847                | 1,152          |
| 64       | 10      | 7,981                | 1,152          |

Default configuration; `model.count_parameters()` gives the trainable count.

## Usage

```python
import torch
from torch import nn
from grfnet import GRFNet, channel_positions

# Classification: raw scores for CrossEntropyLoss
model = GRFNet(n_channels=22, n_times=1000, n_classes=4, sfreq=250.0)
loss = nn.functional.cross_entropy(model(torch.randn(8, 22, 1000)), torch.randint(0, 4, (8,)))

# Regression or multi-label: the loss picks the task
regressor = GRFNet(n_channels=8, n_outputs=1)

# Embeddings and custom heads
encoder = GRFNet(n_channels=8, head=None)                      # forward returns the embedding
mlp = GRFNet(n_channels=8, head=lambda dim: nn.Sequential(nn.Linear(dim, 32), nn.ReLU(), nn.Linear(32, 3)))

# Kernel bank: explicit dilations and a fixed seed for the frozen kernels
model = GRFNet(n_channels=8, n_classes=3, dilations=[1, 3, 9, 27], bank_seed=0)

# Pooling statistics: built-in names or functions (batch, features, time) -> (batch, features)
model = GRFNet(n_channels=8, n_classes=2, statistics=("ppv", "logvar", lambda f: f.amax(-1) - f.amin(-1)))

# Geometry-seeded spatial filters from coordinates in any dimension, or 10-20 names (eeg extra)
model = GRFNet(n_channels=4, n_classes=2, channel_coords=[[0, 1], [1, 0], [0, -1], [-1, 0]])
model = GRFNet(n_channels=4, n_classes=2, channel_names=["C3", "Cz", "C4", "Pz"])
model = GRFNet(n_channels=3, n_classes=2, channel_coords=channel_positions(["C3", "Cz", "C4"], ndim=3))
```

**Dilations.** The default bank uses dilations `(1, 2, 4, 8)`. With `sfreq`,
these are treated as values for 250 Hz and rescaled, keeping each kernel's span
in seconds roughly constant: 1000 Hz gives `[4, 8, 16, 32]`. `dilations` is used
as given.

**Statistics.** `"ppv"` is a differentiable proportion of positive values with a
learnable threshold per feature, `"logvar"` (alias `"bandpower"`) the
log-variance, then `"mean"`, `"max"` and `"std"`.

**Geometry seeding.** Spatial filters start as smooth patterns over the sensor
positions: a gradient per axis, the squared radius, products of axes and
differences of squared axes. Duplicate and constant patterns are skipped, and
remaining components start random. `model.spatial.n_geometry_components` gives
the number seeded.

## Configuration

| Argument              | Default                     | Description                                                                         |
| --------------------- | --------------------------- | ----------------------------------------------------------------------------------- |
| `n_channels`          | required                    | Input channels.                                                                     |
| `n_times`             | `None`                      | Expected length; recorded only.                                                     |
| `n_outputs`           | `None`                      | Linear head size; `n_classes` is an alias.                                          |
| `head`                | `"linear"`                  | `"linear"`, `None`, an `nn.Module`, or a function of `embedding_dim` returning one. |
| `sfreq`               | `None`                      | Sampling rate; rescales `base_dilations` from 250 Hz.                               |
| `spatial_components`  | `16`                        | Learned spatial components.                                                         |
| `spatial_init`        | `None`                      | `"random"` or `"geometry"`; `None` picks geometry when positions are given.         |
| `channel_names`       | `None`                      | 10-20 electrode names (eeg extra).                                                  |
| `channel_coords`      | `None`                      | Sensor coordinates `(n_channels, d)`.                                               |
| `kernel_length`       | `9`                         | Kernel length; odd.                                                                 |
| `base_dilations`      | `None`                      | Dilations before rescaling; `None` is `(1, 2, 4, 8)`.                               |
| `dilations`           | `None`                      | Dilations used as given; excludes `sfreq` and `base_dilations`.                     |
| `fixed_per_group`     | `2`                         | Frozen kernels per component and dilation.                                          |
| `learnable_per_group` | `1`                         | Learnable kernels per component and dilation; `0` freezes the whole bank.           |
| `statistics`          | `("ppv", "logvar", "mean")` | Pooling statistics: names or functions.                                             |
| `ppv_tau`, `ppv_bias` | `0.5`, `True`               | PPV temperature and learnable threshold.                                            |
| `logvar_eps`          | `1e-6`                      | Variance floor for `"logvar"` and `"std"`.                                          |
| `dropout`             | `0.3`                       | Dropout before the head.                                                            |
| `bank_seed`           | `None`                      | Seed for the frozen kernels; `None` uses the global generator.                      |

## Training and evaluation

Any PyTorch training loop works. For the EEGMMIDB benchmark and its results, see
the [repository README](../../../../README.md#results) and the
[experiments README](../../experiments/README.md). The baseline config uses
`dilations = [1, 2, 3, 5]` at 160 Hz.

## Ablations

EEGMMIDB imagined left versus right fist, 109 subjects, 5 subject-disjoint
folds, one seed per run. Each row changes one setting of its base. *Pooled* is
test balanced accuracy over all trials; *Val* is the mean validation balanced
accuracy of the selected epochs; *+/−* counts subjects that improved or
declined; *p* is a two-sided Wilcoxon signed-rank test over subjects. Run names
are `grfnet_eegmmidb_imagery_lr_<variant>`.

**Normalisation and filtering** (base: baseline config, 0.698)

| Variant      | Change                         | Pooled | Δ      | +/−   | p        | Val   |
| ------------ | ------------------------------ | ------ | ------ | ----- | -------- | ----- |
| `ea`         | Euclidean alignment            | 0.742  | +0.044 | 66/40 | 1.3e-4   | 0.764 |
| `ea_bp_8_30` | Euclidean alignment, 8–30 Hz   | 0.713  | +0.015 | 63/46 | 0.24     | 0.741 |
| `bp_8_30`    | 8–30 Hz band-pass              | 0.638  | −0.059 | 32/75 | 4.0e-5   | 0.683 |
| `geometry`   | Geometry-seeded spatial filters | 0.693 | −0.004 | 46/54 | 0.34     | 0.730 |
| `seed1`      | Seed 1                         | 0.704  | +0.007 | 57/51 | 0.34     | 0.734 |
| `seed2`      | Seed 2                         | 0.715  | +0.017 | 65/40 | 0.017    | 0.757 |

**One-setting ablations** (base: `ea`, 0.742, chosen by validation score)

| Variant                 | Change                                  | Pooled | Δ      | +/−   | p       | Val   |
| ----------------------- | --------------------------------------- | ------ | ------ | ----- | ------- | ----- |
| `ea_win_0p5_4`          | Window 0.5–4 s                          | 0.813  | +0.072 | 87/21 | 6.9e-12 | 0.826 |
| `ea_win_0p5_2p5`        | Window 0.5–2.5 s                        | 0.804  | +0.063 | 81/27 | 2.0e-9  | 0.810 |
| `ea_win_0_2`            | Window 0–2 s                            | 0.794  | +0.053 | 79/28 | 8.8e-9  | 0.816 |
| `ea_stats_all`          | Statistics ppv, logvar, mean, max, std  | 0.789  | +0.048 | 79/28 | 4.6e-8  | 0.800 |
| `ea_dil_1_2_4_8`        | Dilations 1, 2, 4, 8                    | 0.763  | +0.021 | 64/39 | 1.0e-3  | 0.775 |
| `ea_klen15`             | Kernel length 15                        | 0.762  | +0.020 | 68/36 | 1.2e-3  | 0.778 |
| `ea_dil_1_3_5_7`        | Dilations 1, 3, 5, 7                    | 0.759  | +0.018 | 67/34 | 1.1e-3  | 0.763 |
| `ea_ep200`              | 200 epochs, patience 40                 | 0.751  | +0.009 | 52/38 | 0.032   | 0.776 |
| `ea_dil_1_2_3_5_8`      | Dilations 1, 2, 3, 5, 8                 | 0.750  | +0.008 | 64/41 | 0.12    | 0.768 |
| `ea_no_early_stop`      | Patience 100                            | 0.748  | +0.007 | 25/14 | 0.081   | 0.765 |
| `ea_lr3em3`             | Learning rate 0.003                     | 0.746  | +0.005 | 47/55 | 0.82    | 0.768 |
| `ea_no_sched`           | No learning-rate schedule               | 0.746  | +0.005 | 42/35 | 0.30    | 0.776 |
| `ea_bs32`               | Batch size 32                           | 0.745  | +0.003 | 50/47 | 0.66    | 0.765 |
| `ea_drop0p5`            | Dropout 0.5                             | 0.745  | +0.004 | 51/49 | 0.57    | 0.765 |
| `ea_spatial8`           | 8 spatial filters                       | 0.745  | +0.004 | 58/49 | 0.57    | 0.754 |
| `ea_stats_logvar`       | Statistics logvar                       | 0.744  | +0.003 | 51/57 | 0.67    | 0.757 |
| `ea_stats_ppv_logvar`   | Statistics ppv, logvar                  | 0.744  | +0.003 | 57/49 | 0.50    | 0.759 |
| `ea_tau1`               | PPV temperature 1.0                     | 0.743  | +0.002 | 39/28 | 0.55    | 0.764 |
| `ea_spatial24`          | 24 spatial filters                      | 0.742  | +0.000 | 57/49 | 0.99    | 0.749 |
| `ea_wd0`                | Weight decay 0                          | 0.742  | +0.000 | 1/0   | 0.32    | 0.764 |
| `ea_wd1em3`             | Weight decay 0.001                      | 0.742  | +0.000 | 2/1   | 0.59    | 0.764 |
| `ea_fixed4`             | 4 frozen kernels per group              | 0.740  | −0.001 | 57/48 | 0.99    | 0.756 |
| `ea_tau0p25`            | PPV temperature 0.25                    | 0.740  | −0.001 | 36/35 | 0.77    | 0.761 |
| `ea_klen7`              | Kernel length 7                         | 0.740  | −0.002 | 43/56 | 0.41    | 0.752 |
| `ea_spatial32`          | 32 spatial filters                      | 0.738  | −0.004 | 49/55 | 0.68    | 0.761 |
| `ea_val0p2`             | Validation share 0.2                    | 0.736  | −0.005 | 49/54 | 0.35    | 0.755 |
| `ea_drop0p1`            | Dropout 0.1                             | 0.735  | −0.006 | 44/53 | 0.18    | 0.754 |
| `ea_learn2`             | 2 learnable kernels per group           | 0.735  | −0.006 | 46/57 | 0.22    | 0.743 |
| `ea_bs128`              | Batch size 128                          | 0.734  | −0.008 | 46/56 | 0.18    | 0.758 |
| `ea_ep50`               | 50 epochs                               | 0.732  | −0.009 | 42/55 | 0.037   | 0.741 |
| `ea_bp_4_40`            | 4–40 Hz band-pass                       | 0.723  | −0.018 | 43/66 | 0.062   | 0.752 |
| `ea_lr3em4`             | Learning rate 0.0003                    | 0.716  | −0.026 | 34/67 | 2.3e-5  | 0.723 |
| `ea_bp_8_30`            | 8–30 Hz band-pass (stage 1)             | 0.713  | −0.029 | 41/67 | 6.8e-3  | 0.741 |
| `ea_motor17`            | 17 motor channels                       | 0.685  | −0.056 | 29/76 | 4.0e-7  | 0.701 |
| `ea_bp_8_13`            | 8–13 Hz band-pass                       | 0.669  | −0.073 | 31/77 | 1.1e-7  | 0.692 |
| `ea_motor17_geometry`   | 17 motor channels, geometry seeding     | 0.665  | −0.077 | 28/80 | 2.1e-10 | 0.691 |

**Combined settings** (`ea_best`: alignment, 0.5–4 s window, statistics
ppv, logvar, mean, max, std, dilations 1, 2, 4, 8; 4,135 trainable parameters,
3.1 MFLOPs per trial)

| Run                | Seed | Pooled | Per fold, mean (SD) | Per subject, mean (SD) | Cohen's kappa | Val   |
| ------------------ | ---- | ------ | ------------------- | ---------------------- | ------------- | ----- |
| `ea_best`          | 0    | 0.806  | 0.806 (0.016)       | 0.806 (0.129)          | 0.611         | 0.827 |
| `ea_best_seed1`    | 1    | 0.811  | 0.812 (0.009)       | 0.812 (0.122)          | 0.623         | 0.803 |
| `ea_best_seed2`    | 2    | 0.808  | 0.808 (0.040)       | 0.808 (0.127)          | 0.616         | 0.858 |
| Mean (SD) of seeds |      | 0.808 (0.003) | 0.809 (0.024) over 15 folds | 0.809 (0.123), seed-averaged | 0.617 | |

Against `ea`, the seed-averaged per-subject accuracy rises by 0.067 (85
subjects improved, 24 declined, p = 4.5e-10). Against the window change alone
(`ea_win_0p5_4`, 0.813) it changes by −0.004 (50/56, p = 0.49): with the
window, max pooling and the wider dilations add nothing measurable. The gate
weights logvar (0.21–0.40) and mean (0.31–0.35) highest, then max (0.13–0.19).

**Final configuration** (`ea_win_0p5_4`: alignment and 0.5–4 s window, default
architecture; 3,365 trainable parameters, 3.1 MFLOPs per trial)

| Run                       | Seed | Pooled | Per fold, mean (SD) | Per subject, mean (SD) | Cohen's kappa | Val   |
| ------------------------- | ---- | ------ | ------------------- | ---------------------- | ------------- | ----- |
| `ea_win_0p5_4`            | 0    | 0.813  | 0.814 (0.016)       | 0.813 (0.115)          | 0.627         | 0.826 |
| `ea_win_0p5_4_seed1`      | 1    | 0.807  | 0.807 (0.013)       | 0.807 (0.126)          | 0.614         | 0.810 |
| `ea_win_0p5_4_seed2`      | 2    | 0.796  | 0.795 (0.034)       | 0.795 (0.125)          | 0.591         | 0.862 |
| Mean (SD) of seeds        |      | 0.805 (0.009) | 0.805 (0.023) over 15 folds | 0.805 (0.118), seed-averaged | 0.611 | 0.833 |

Against the combined settings, seed-averaged per-subject accuracy differs by
−0.004 (47 better, 60 worse, p = 0.28); against the baseline's three-seed
average it rises by 0.098 (91/17, p = 2.4e-13). The two are equivalent, and the
simpler one is adopted. Its gate weights mean (0.42–0.46) and logvar
(0.42–0.47) almost equally, and ppv least (0.08–0.13). Reproduce with:

```bash
python -m grfnet.train configs/grfnet_eegmmidb_imagery_lr.toml --set "data.normalize=euclidean" --set "data.tmin=0.5" --set "variant=ea_win_0p5_4"
```

**Other tasks** (`ea` settings; run names `grfnet_eegmmidb_<task>_ea`)

| Task                        | Classes | Chance | Pooled | Per subject, mean (SD) | Cohen's kappa |
| --------------------------- | ------- | ------ | ------ | ---------------------- | ------------- |
| `execution_lr`              | 2       | 0.50   | 0.748  | 0.749 (0.112)          | 0.497         |
| `imagery_lr`                | 2       | 0.50   | 0.742  | 0.742 (0.123)          | 0.483         |
| `execution_ff`              | 2       | 0.50   | 0.716  | 0.715 (0.142)          | 0.431         |
| `imagery_ff`                | 2       | 0.50   | 0.658  | 0.660 (0.132)          | 0.316         |
| `imagery_lr`, with rest     | 3       | 0.33   | 0.589  | 0.590 (0.131)          | 0.385         |
| `imagery`                   | 4       | 0.25   | 0.499  | 0.500 (0.130)          | 0.332         |

Findings:

- **Seed noise.** Three seeds of the baseline give 0.706 ± 0.009, and seed 2
  differs from seed 0 at p = 0.017. Differences below about 0.02, or p values
  near 0.05 from a single seed, are not reliable. Over the 35 one-setting
  ablations the Bonferroni threshold is p < 0.0014; the windows, max pooling,
  longer receptive fields, learning rate 0.0003, motor channels and the 8–13 Hz
  band-pass pass it.
- **Largest gains.** Euclidean alignment, dropping the first 0.5 s after the
  cue, and adding max (and std) pooling each add 0.04–0.07. Longer receptive
  fields (dilations up to 7–8, or kernel length 15) add about 0.02. Combined,
  they reach 0.808 ± 0.003 over three seeds, no more than the window alone
  (0.805 ± 0.009): the gains overlap rather than add. The final configuration
  is alignment and the window with the default architecture.
- **Max pooling.** With all five statistics, the gate weights max second after
  logvar (0.20–0.26). Removing ppv and mean alone changes nothing.
- **Band-pass filters and motor channels hurt.** Signal outside 8–30 Hz and
  outside the motor strip contributes, which may include ocular or muscle
  activity.
- **Insensitive.** Spatial filter count, kernel counts, PPV temperature,
  dropout, weight decay, batch size and the schedule change accuracy by less
  than the seed noise.

## Limitations

- Rescaling the default dilations to low rates repeats a dilation: 125 Hz gives
  `[1, 1, 2, 4]` and 160 Hz `[1, 1, 3, 5]`. Pass `dilations` at such rates.
- Few sensors allow few distinct geometry patterns.
- Custom heads and statistics cannot be stored in a config or compiled with
  TorchScript.
- Models built from `channel_names` need the eeg extra to load.

## References

1. Dempster, A., Petitjean, F. and Webb, G. I. (2020). ROCKET: exceptionally fast and accurate time series classification using random convolutional kernels. *Data Mining and Knowledge Discovery*, 34, 1454-1495.
2. Dempster, A., Schmidt, D. F. and Webb, G. I. (2021). MiniRocket: a very fast (almost) deterministic transform for time series classification. *KDD*.
3. Ang, K. K. et al. (2008). Filter Bank Common Spatial Pattern (FBCSP) in brain-computer interface. *IJCNN*.
4. Schirrmeister, R. T. et al. (2017). Deep learning with convolutional neural networks for EEG decoding and visualization. *Human Brain Mapping*, 38(11), 5391-5420.
