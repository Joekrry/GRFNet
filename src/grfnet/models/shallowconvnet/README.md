# ShallowConvNet

Baseline for GRF-Net. The Shallow ConvNet of Schirrmeister et al. [1] as
implemented in braindecode (`ShallowFBCSPNet`), wrapped so it trains,
checkpoints and reports through the same pipeline as GRFNet. Needs the
`baselines` extra (braindecode 2.0+, Python 3.11+, PyTorch 2.6+).

## Architecture

A temporal convolution (40 filters, 25 samples) and a spatial convolution
(40 filters) with batch normalisation, then squaring, average pooling (75
samples, stride 15), a logarithm, dropout 0.5 and a convolutional classifier.
The squaring, pooling and logarithm compute log band power, mirroring filter
bank common spatial patterns. The kernel lengths were set for 250 Hz in the
original paper.

## Usage

```python
import torch
from grfnet import ShallowConvNet, load_model

model = ShallowConvNet(n_channels=64, n_times=640, n_classes=2)   # braindecode defaults
logits = model(torch.randn(8, 64, 640))                           # (batch, n_classes), raw scores

model.save("shallow.pt")
model = load_model("shallow.pt")
```

Keyword arguments go to `braindecode.models.ShallowFBCSPNet` and must be plain
values to be stored in a checkpoint. In an experiment config, set
`name = "ShallowConvNet"` in `[model]`; see
[`configs/shallowconvnet_eegmmidb_imagery_lr.toml`](../../../../configs/shallowconvnet_eegmmidb_imagery_lr.toml).

## Protocol

The baseline uses braindecode's defaults and GRF-Net's training protocol
(AdamW, cosine decay, validation-based epoch selection). Results may differ
from those reported with the original training recipe.

## Results

EEGMMIDB imagined left versus right fist, 109 subjects, 5 subject-disjoint
folds, seeds 0–2, braindecode defaults. *Baseline input*: global scaling,
0–4 s. *Final input*: GRFNet's final configuration, Euclidean alignment and
0.5–4 s. Run names are `shallowconvnet_eegmmidb_imagery_lr[_ea_win_0p5_4][_seed<n>]`.

| Input    | Seed | Pooled | Per fold, mean (SD) | Per subject, mean (SD) | Cohen's kappa | Val   |
| -------- | ---- | ------ | ------------------- | ---------------------- | ------------- | ----- |
| Baseline | 0    | 0.818  | 0.819 (0.018)       | 0.819 (0.122)          | 0.636         | 0.834 |
| Baseline | 1    | 0.823  | 0.824 (0.018)       | 0.824 (0.116)          | 0.647         | 0.834 |
| Baseline | 2    | 0.815  | 0.815 (0.037)       | 0.816 (0.127)          | 0.630         | 0.859 |
| Final    | 0    | 0.737  | 0.737 (0.023)       | 0.737 (0.126)          | 0.473         | 0.779 |
| Final    | 1    | 0.735  | 0.735 (0.022)       | 0.735 (0.123)          | 0.470         | 0.764 |
| Final    | 2    | 0.715  | 0.715 (0.037)       | 0.716 (0.132)          | 0.430         | 0.802 |

Comparisons with GRFNet use seed-averaged per-subject balanced accuracy and
paired Wilcoxon signed-rank tests over 109 subjects. The rule fixed before the
runs compared models on the same input, with p < 0.0125 (four comparisons) and
a difference above 0.017.

| Comparison                                   | GRFNet − ShallowConvNet | GRFNet better / worse | p       |
| -------------------------------------------- | ----------------------- | --------------------- | ------- |
| Baseline input (rule)                        | −0.112                  | 16 / 93               | 1.5e-15 |
| Final input (rule)                           | +0.076                  | 87 / 22               | 1.5e-13 |
| Each at its best input (exploratory)         | −0.015                  | 50 / 58               | 0.11    |

- **Best accuracy:** ShallowConvNet scores 0.819 (SD 0.004) on the baseline
  input, with 106,482 trainable parameters and 79 MFLOPs per 4 s trial.
- **Effect of GRFNet's final input:** it lowers ShallowConvNet's accuracy by
  0.091 (p = 6e-14). Training accuracy at the selected epoch reaches 0.99
  against 0.91 on the baseline input, so it overfits more.
- **Against GRFNet:** compared at each model's best input, ShallowConvNet and
  GRFNet do not differ by the rule (−0.015, below the 0.017 floor). GRFNet uses
  32 times fewer trainable parameters and about 25 times fewer operations.

## References

1. Schirrmeister, R. T. et al. (2017). Deep learning with convolutional neural networks for EEG decoding and visualization. *Human Brain Mapping*, 38(11), 5391-5420.
