# EEGNet

Baseline for GRF-Net. EEGNet-8,2 [1] as implemented in braindecode [2], wrapped
so it trains, checkpoints and reports through the same pipeline as GRFNet.
Needs the `baselines` extra (braindecode 2.0+, Python 3.11+, PyTorch 2.6+).

## Architecture

A temporal convolution (8 filters, 64 samples), a depthwise spatial
convolution (depth multiplier 2) with batch normalisation, ELU and average
pooling, then a separable convolution (16 filters), pooling, dropout 0.25 and
a linear classifier. The kernel lengths were set for 128 Hz in the original
paper.

## Usage

```python
import torch
from grfnet import EEGNet, load_model

model = EEGNet(n_channels=64, n_times=640, n_classes=2)   # braindecode defaults
logits = model(torch.randn(8, 64, 640))                   # (batch, n_classes), raw scores

model = EEGNet(n_channels=64, n_times=640, n_classes=2, kernel_length=80, drop_prob=0.5)
model.save("eegnet.pt")
model = load_model("eegnet.pt")                           # any GRF-Net or baseline checkpoint
```

Keyword arguments go to `braindecode.models.EEGNet` and must be plain values to
be stored in a checkpoint. In an experiment config, set `name = "EEGNet"` in
`[model]` and put the arguments in `[model.kwargs]`; see
[`configs/eegnet_eegmmidb_imagery_lr.toml`](../../../../configs/eegnet_eegmmidb_imagery_lr.toml).

## Protocol

The baseline uses braindecode's defaults and GRF-Net's training protocol
(AdamW, cosine decay, validation-based epoch selection). Results may differ
from those reported with the original training recipe.

## Results

EEGMMIDB imagined left versus right fist, 109 subjects, 5 subject-disjoint
folds, seeds 0–2, braindecode defaults. *Baseline input*: global scaling,
0–4 s. *Final input*: GRFNet's final configuration, Euclidean alignment and
0.5–4 s. Run names are `eegnet_eegmmidb_imagery_lr[_ea_win_0p5_4][_seed<n>]`.

| Input    | Seed | Pooled | Per fold, mean (SD) | Per subject, mean (SD) | Cohen's kappa | Val   |
| -------- | ---- | ------ | ------------------- | ---------------------- | ------------- | ----- |
| Baseline | 0    | 0.821  | 0.821 (0.011)       | 0.822 (0.122)          | 0.642         | 0.833 |
| Baseline | 1    | 0.828  | 0.828 (0.023)       | 0.829 (0.119)          | 0.656         | 0.832 |
| Baseline | 2    | 0.827  | 0.827 (0.031)       | 0.828 (0.121)          | 0.654         | 0.861 |
| Final    | 0    | 0.778  | 0.778 (0.014)       | 0.779 (0.133)          | 0.556         | 0.799 |
| Final    | 1    | 0.783  | 0.783 (0.015)       | 0.784 (0.136)          | 0.566         | 0.797 |
| Final    | 2    | 0.783  | 0.783 (0.033)       | 0.784 (0.131)          | 0.566         | 0.815 |

Comparisons with GRFNet use seed-averaged per-subject balanced accuracy and
paired Wilcoxon signed-rank tests over 109 subjects. The rule fixed before the
runs compared models on the same input, with p < 0.0125 (four comparisons) and
a difference above 0.017.

| Comparison                                   | GRFNet − EEGNet | GRFNet better / worse | p       |
| -------------------------------------------- | --------------- | --------------------- | ------- |
| Baseline input (rule)                        | −0.118          | 16 / 93               | 3e-15   |
| Final input (rule)                           | +0.023          | 65 / 43               | 0.0038  |
| Each at its best input (exploratory)         | −0.021          | 36 / 73               | 0.0005  |

- **Best accuracy:** EEGNet scores 0.825 (SD 0.004) on the baseline input,
  with 2,770 trainable parameters and 43 MFLOPs per 4 s trial. It is the most
  accurate model in this repository on EEGMMIDB left versus right imagery.
- **Effect of GRFNet's final input:** it lowers EEGNet's accuracy by 0.044
  (p = 1.2e-10). On that input GRFNet is ahead. Compared at each model's best
  input, EEGNet leads GRFNet (0.805) by 0.021.
- **Cost:** GRFNet needs 3.1 MFLOPs per trial on its final input, about 14 times
  fewer than EEGNet at its best input.

## References

1. Lawhern, V. J. et al. (2018). EEGNet: a compact convolutional neural network for EEG-based brain-computer interfaces. *Journal of Neural Engineering*, 15(5), 056013.
2. Schirrmeister, R. T. et al. (2017). Deep learning with convolutional neural networks for EEG decoding and visualization. *Human Brain Mapping*, 38(11), 5391-5420.
