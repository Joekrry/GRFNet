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

## References

1. Schirrmeister, R. T. et al. (2017). Deep learning with convolutional neural networks for EEG decoding and visualization. *Human Brain Mapping*, 38(11), 5391-5420.
