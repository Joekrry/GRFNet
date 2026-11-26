# FastGRFNet

GRF-Net with fast weights [1, 2] in its spatial front end. The spatial filters
change from trial to trial within one subject's sequence of trials, so the
model can adapt to a new user without retraining. All other layers are
GRF-Net's.

## Architecture

The spatial filters are `W_slow + ΔW`. `W_slow` is GRF-Net's 1x1 spatial
convolution, with the same initialisation and optional geometry seeding.
`ΔW` (spatial components x channels) is the state, zero at the start of each
subject's sequence. After each trial is predicted, a writer network maps a
summary of the trial to a key `k`, a value `v` and a rate `η`, and `ΔW` is
updated by the delta rule:

```text
ΔW <- λ ΔW + η (v − ΔW k) kᵀ,    then ‖ΔW‖_F clipped to delta_clip
```

- **Writer input.** The log-variance of each input channel, standardised
  across channels. In `label_feedback` mode, the trial's one-hot label and a
  known-label flag are appended.
- **Key and rate.** `k` has unit norm. `η = eta_max · sigmoid(·)`.
- **Starting point.** The value projection is initialised to zero, so the
  state stays at zero and a new FastGRFNet behaves exactly like GRFNet.

| Write mode       | Writer input                       | Labels at test time               |
| ---------------- | ---------------------------------- | --------------------------------- |
| `off`            | none; never writes                 | no (reproduces GRFNet exactly)    |
| `unsupervised`   | channel log-variance summary       | no                                |
| `label_feedback` | summary plus the trial's label     | yes, after that trial's prediction |

The label is used only by the write, after the trial's logits are computed, so
it never affects the prediction it belongs to. `label_feedback` is an online
supervised regime and is reported separately from `off` and `unsupervised`.

## Usage

```python
import torch
from grfnet import FastGRFNet

model = FastGRFNet(n_channels=64, n_classes=2, dilations=[1, 2, 3, 5], write_mode="unsupervised")
logits, state = model(torch.randn(1, 64, 560))     # one trial, zero state: as GRFNet

state = model.init_state(batch_size=1)             # stream one subject's trials in order
for trial in torch.randn(10, 1, 64, 560):
    logits, state = model(trial, state)

model = FastGRFNet(n_channels=64, n_classes=2, write_mode="label_feedback")
logits, state = model(torch.randn(4, 64, 560), state=None, label=torch.tensor([0, 1, -1, 1]))  # -1: unknown
```

Each batch element is a separate sequence with its own state. Checkpoints use
GRF-Net's format (`model.save`, `FastGRFNet.load`). TorchScript is not
supported, because the model is stateful.

## Training

The writer learns only from sequences of trials. Experiments train FastGRFNet
with episodic meta-training (`training.mode = "episodic"`): each training
subject's trials in recording order from a zero state, several subjects in
parallel, with back-propagation through the writes over chunks of
`chunk_length` trials. Validation and test subjects are streamed the same way.
See the [experiments README](../../experiments/README.md#protocol) and
[`configs/fastgrfnet_eegmmidb_imagery_lr.toml`](../../../../configs/fastgrfnet_eegmmidb_imagery_lr.toml).

```bash
python -m grfnet.train configs/fastgrfnet_eegmmidb_imagery_lr.toml
python -m grfnet.train configs/fastgrfnet_eegmmidb_imagery_lr.toml --set model.kwargs.write_mode=label_feedback --set variant=label_feedback
```

The run report's `adaptation.png` shows test accuracy by trial position.

## Configuration

All arguments of [GRFNet](../grfnet/README.md#configuration), plus:

| Argument        | Default          | Meaning                                                   |
| --------------- | ---------------- | --------------------------------------------------------- |
| `write_mode`    | `"unsupervised"` | `"off"`, `"unsupervised"` or `"label_feedback"`.          |
| `decay`         | `0.95`           | Fast-weight decay `λ` per trial, in (0, 1).               |
| `eta_max`       | `0.5`            | Largest write rate `η`.                                   |
| `delta_clip`    | `1.0`            | Largest Frobenius norm of `ΔW`.                           |
| `writer_hidden` | `32`             | Hidden units of the writer.                               |

The writer adds 4,753 trainable parameters in `unsupervised` mode and 4,849 in
`label_feedback` mode for 64 channels, 16 spatial components and two classes,
on top of GRF-Net's 3,365.

## Results

EEGMMIDB imagined left versus right fist, 109 subjects, 5 subject-disjoint
folds, Euclidean alignment and a 0.5–4 s window (GRF-Net's final input),
episodic training, seeds 0–2. Run names are
`fastgrfnet_eegmmidb_imagery_lr_ea_win_0p5_4_<mode>[_seed<n>]`.

| Write mode       | Seed | Pooled | Per fold, mean (SD) | Per subject, mean (SD) | Cohen's kappa | Val   |
| ---------------- | ---- | ------ | ------------------- | ---------------------- | ------------- | ----- |
| `off`            | 0    | 0.805  | 0.805 (0.024)       | 0.805 (0.121)          | 0.610         | 0.808 |
| `off`            | 1    | 0.802  | 0.802 (0.014)       | 0.802 (0.120)          | 0.605         | 0.796 |
| `off`            | 2    | 0.798  | 0.798 (0.038)       | 0.798 (0.118)          | 0.596         | 0.861 |
| `unsupervised`   | 0    | 0.742  | 0.742 (0.026)       | 0.742 (0.128)          | 0.484         | 0.767 |
| `unsupervised`   | 1    | 0.738  | 0.738 (0.014)       | 0.738 (0.131)          | 0.476         | 0.757 |
| `unsupervised`   | 2    | 0.731  | 0.731 (0.031)       | 0.731 (0.120)          | 0.463         | 0.802 |
| `label_feedback` | 0    | 0.841  | 0.841 (0.006)       | 0.842 (0.079)          | 0.682         | 0.855 |
| `label_feedback` | 1    | 0.833  | 0.833 (0.020)       | 0.835 (0.071)          | 0.667         | 0.852 |
| `label_feedback` | 2    | 0.827  | 0.827 (0.018)       | 0.827 (0.079)          | 0.653         | 0.866 |

Comparisons use seed-averaged per-subject balanced accuracy, with paired
Wilcoxon signed-rank tests over 109 subjects.

| Comparison                      | Difference | Better / worse | p       |
| ------------------------------- | ---------- | -------------- | ------- |
| `off` vs GRFNet, final config   | −0.003     | 52 / 56        | 0.50    |
| `unsupervised` vs `off`         | −0.065     | 6 / 103        | 6e-18   |
| `label_feedback` vs `off`       | +0.033     | 69 / 40        | 5.7e-5  |

- **Episodic training costs nothing.** With writes off, FastGRFNet matches
  GRFNet trained the standard way (0.802 against 0.805).
- **Unsupervised writes hurt.** Every seed loses about 0.065, and only 6
  subjects improve. Second-half accuracy gains no more than with writes off
  (+0.010 against +0.016, p = 0.38), so there is no sign of adaptation.
- **The label-feedback gain comes from the order of the cues, not from
  adaptation.** In EEGMMIDB's imagery runs, consecutive trials have opposite
  labels 74% of the time, so guessing the opposite of the previous label scores
  0.741 without any EEG. With label feedback, accuracy zigzags with trial
  position: 0.752, 0.982, 0.789, 0.936, 0.780 and 0.890 at positions 0–5,
  against 0.78, 0.80, 0.77, 0.74, 0.81 and 0.73 with writes off. Accuracy is
  also lower in the second half of each sequence than in the first (0.825
  against 0.842), the opposite of adaptation. The writer has learned the cue
  sequence. The +0.033 therefore does not show that fast weights adapt to a
  subject's EEG.

## Limitations

- Fast weights only help if the writer is trained on sequences of trials,
  which is why FastGRFNet trains only in episodic mode.
- EEGMMIDB has about 45 left/right imagery trials per subject, so each
  adaptation sequence is short.
- On data whose cue order is predictable, `label_feedback` can learn the label
  sequence instead of the EEG. A fair test needs trials presented in random
  order, or a dataset with random cue order.

## References

1. Schmidhuber, J. (1992). Learning to control fast-weight memories: an alternative to dynamic recurrent networks. *Neural Computation*, 4(1), 131-139.
2. Schlag, I., Irie, K. and Schmidhuber, J. (2021). Linear transformers are secretly fast weight programmers. *ICML*, 9355-9366.
