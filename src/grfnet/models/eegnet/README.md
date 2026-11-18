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

## References

1. Lawhern, V. J. et al. (2018). EEGNet: a compact convolutional neural network for EEG-based brain-computer interfaces. *Journal of Neural Engineering*, 15(5), 056013.
2. Schirrmeister, R. T. et al. (2017). Deep learning with convolutional neural networks for EEG decoding and visualization. *Human Brain Mapping*, 38(11), 5391-5420.
