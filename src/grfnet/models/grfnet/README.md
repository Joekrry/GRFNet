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
