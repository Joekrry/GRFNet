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

## Results

EEGMMIDB imagined left versus right fist, 109 subjects, 5 subject-disjoint
folds, Euclidean alignment and a 0.5–4 s window, seeds 0–2. Run names are
`grfnetseq_eegmmidb_imagery_lr_ea_win_0p5_4[_seed<n>]`. Trials are scored by
the prediction at the last time step.

| Seed | Pooled | Per fold, mean (SD) | Per subject, mean (SD) | Cohen's kappa | Val   |
| ---- | ------ | ------------------- | ---------------------- | ------------- | ----- |
| 0    | 0.623  | 0.624 (0.025)       | 0.623 (0.096)          | 0.246         | 0.652 |
| 1    | 0.611  | 0.611 (0.011)       | 0.610 (0.091)          | 0.222         | 0.648 |
| 2    | 0.614  | 0.613 (0.026)       | 0.613 (0.100)          | 0.227         | 0.666 |

- **Against GRFNet's final configuration:** the mean over seeds is 0.616
  (SD 0.006), against 0.805. Seed-averaged per-subject accuracy is 0.190 lower,
  with 4 subjects better and 105 worse (p = 2.4e-19). Six subjects are below
  chance.
- **Accuracy over the trial:** in the report for seed 0, fold 0,
  `time_course.png` shows accuracy of 0.72–0.77 between 0.5 and 1.0 s after the
  cue (peak 0.767 at 0.93 s). It then falls to about 0.62, and the last step
  scores 0.602.
  - **The decision rule loses information:** scoring by the last step,
    decided before the runs, discards the most informative part of the trial.
  - **The early accuracy may not be motor imagery:** it appears within a few
    samples of the window opening, which suggests a slow lateralised signal,
    such as eye position following the cue, rather than mu or beta rhythms.
    The peak was found on test data and is not a valid performance estimate.
- **Cost:** 3,365 trainable parameters and 4.4 MFLOPs for a 3.5 s trial
  (logits at every step). Latency is 3.4 ms on the CPU for one trial.

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
