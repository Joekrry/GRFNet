# GRFNetSeq

GRF-Net made causal, for streaming use. It predicts at every time step, and
each prediction depends only on the current and earlier samples. A recording
can be processed chunk by chunk as it arrives, with the same outputs as a
single pass.

## Architecture

The layers are GRF-Net's, with two changes:

- **Causal kernel bank.** Each dilated convolution is padded on the left only,
  so its output at time `t` uses samples up to `t`.
- **Sliding statistics.** PPV, log-variance and mean are computed over the last
  `window` samples, not over the whole trial. Near the start of a recording
  they use all samples so far.

The gated statistics feed the linear head at every time step, giving logits of
shape `(batch, n_classes, n_times)`. `stream` keeps the last `context` input
samples between chunks. `context` is the bank's receptive field plus the
window, less two. In eval mode, streaming in chunks of any size reproduces
`forward` on the whole recording. With the same arguments, GRFNetSeq has as
many trainable parameters as GRFNet (3,365 for 64 channels and two classes).

## Usage

```python
import torch
from grfnet import GRFNetSeq

model = GRFNetSeq(n_channels=64, n_classes=2, dilations=[1, 2, 3, 5], window=64).eval()
logits = model(torch.randn(1, 64, 560))            # (1, 2, 560): a prediction at every sample

buffer = None                                       # stream the same recording in 0.1 s chunks
for chunk in torch.randn(1, 64, 560).split(16, dim=-1):
    logits, buffer = model.stream(chunk, buffer)    # (1, 2, 16) per chunk
```

## Training and evaluation

In experiments, the loss is cross-entropy at every time step against the
trial's label. Each trial is scored by the prediction at its last time step,
which is the decision available once the trial window has ended. The run
report adds `time_course.png`: test accuracy of the prediction at every time
step. See
[`configs/grfnetseq_eegmmidb_imagery_lr.toml`](../../../../configs/grfnetseq_eegmmidb_imagery_lr.toml).

## Configuration

All arguments of [GRFNet](../grfnet/README.md#configuration) except `head`,
`n_outputs`, `statistics` and `ppv_bias`, plus:

| Argument | Default | Meaning                                                  |
| -------- | ------- | -------------------------------------------------------- |
| `window` | `64`    | Samples in the sliding window of each pooling statistic. |

The statistics are PPV, log-variance and mean, the PPV thresholds are learned
and the head is linear.

## Limitations

- In training mode, BatchNorm uses batch statistics. Streaming is exact only in
  eval mode.
- Early time steps see little of the trial, and the per-step loss includes
  them.
