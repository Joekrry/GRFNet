# GRFNet2d

GRF-Net applied to time-frequency images. Each electrode's signal becomes a
log-power spectrogram, and the electrodes are the image channels. The spatial
filters, frozen and learnable kernel bank, pooling statistics and gate follow
GRF-Net's design in two dimensions.

## Architecture

1. **Spectrogram.** Short-time Fourier transform per electrode (Hann window of
   `n_fft` samples, step `hop`, centred frames). The log power is kept up to
   `fmax` Hz, giving `(electrodes, frequency bins, frames)`.
2. **Spatial filters.** A 1x1 convolution mixes the electrodes into
   `spatial_components` maps, followed by batch normalisation. With
   `channel_names` or `channel_coords`, the filters are seeded from electrode
   geometry as in GRFNet.
3. **Kernel bank.** Depthwise 2-D kernels of `kernel_size` (frequency bins x
   frames), dilated along time. Per spatial map and dilation there are
   `fixed_per_group` frozen kernels (random, mean-centred) and
   `learnable_per_group` trained ones.
4. **Pooling and head.** PPV, log-variance and mean over every
   time-frequency position of each feature map, a softmax gate over the three
   statistics, dropout and a linear layer.

With the EEGMMIDB config (64 electrodes, 3.5 s at 160 Hz, `n_fft = 64`,
`hop = 16`, `fmax = 40`), the input image is 64 x 17 x 36. The model has 3,077
trainable and 1,440 frozen parameters for two classes.

## Usage

```python
import torch
from grfnet import GRFNet2d

model = GRFNet2d(n_channels=64, n_classes=2, sfreq=160.0, fmax=40.0)
logits = model(torch.randn(8, 64, 560))      # raw EEG in; the spectrogram is computed inside
image = model.spectrogram(torch.randn(8, 64, 560))   # (8, 64, 17, 36) log power
```

In experiments, set `name = "GRFNet2d"`; the data's sampling rate is passed
as `sfreq`. See
[`configs/grfnet2d_eegmmidb_imagery_lr.toml`](../../../../configs/grfnet2d_eegmmidb_imagery_lr.toml).
The run report draws the stream gates, spatial filter weights, embeddings and
saliency. It skips kernel frequency responses and activation patterns, which
assume filters on the raw signal.

## Results

EEGMMIDB imagined left versus right fist, 109 subjects, 5 subject-disjoint
folds, Euclidean alignment and a 0.5–4 s window, seeds 0–2, with the config's
spectrogram settings. Run names are
`grfnet2d_eegmmidb_imagery_lr_ea_win_0p5_4[_seed<n>]`.

| Seed | Pooled | Per fold, mean (SD) | Per subject, mean (SD) | Cohen's kappa | Val   |
| ---- | ------ | ------------------- | ---------------------- | ------------- | ----- |
| 0    | 0.600  | 0.599 (0.018)       | 0.598 (0.105)          | 0.199         | 0.651 |
| 1    | 0.606  | 0.606 (0.017)       | 0.605 (0.109)          | 0.212         | 0.602 |
| 2    | 0.603  | 0.604 (0.037)       | 0.604 (0.113)          | 0.206         | 0.667 |

- **Against GRFNet's final configuration:** the mean over seeds is 0.603
  (SD 0.003), against 0.805. Seed-averaged per-subject accuracy is 0.203 lower,
  with 5 subjects better and 104 worse (p = 1.1e-18). Twelve subjects are below
  chance.
- **Possible reasons, not tested here:**
  - **Time resolution:** with 0.4 s windows every 0.1 s, the spectrogram has
    coarse time resolution.
  - **Phase:** log power discards phase and slow components, including the
    early lateralised signal GRFNetSeq picks up.
  - **Spatial filtering:** the spatial filters mix log power across electrodes,
    which cannot reproduce the spatial filtering of the raw signal that GRF-Net
    relies on.
- **Cost:** 3,077 trainable and 1,440 frozen parameters, 3.9 MFLOPs per trial.

## Configuration

| Argument               | Default     | Meaning                                                          |
| ---------------------- | ----------- | ---------------------------------------------------------------- |
| `sfreq`                | `None`      | Sampling rate in Hz; needed for `fmax`.                          |
| `n_fft`, `hop`         | `64`, `16`  | STFT window and step in samples.                                 |
| `fmax`                 | `None`      | Highest frequency kept, in Hz; `None` keeps every bin.           |
| `kernel_size`          | `(3, 5)`    | Kernel size in frequency bins and frames; both odd.              |
| `dilations`            | `(1, 2, 4)` | Dilations along time, in frames.                                 |
| `spatial_components`   | `16`        | Spatial maps.                                                    |
| `fixed_per_group`, `learnable_per_group` | `2`, `1` | Frozen and trained kernels per map and dilation.    |
| `spatial_init`, `channel_names`, `channel_coords` | | Geometry seeding, as in GRFNet.                          |
| `ppv_tau`, `logvar_eps`, `dropout`, `bank_seed` | `0.5`, `1e-6`, `0.3`, `None` | As in GRFNet.               |

## Limitations

- Inputs need more than `n_fft // 2` samples.
- The spatial filters mix log power, not the raw signal, so they are not
  spatial filters in the sense of common spatial patterns, and activation
  patterns from the raw signal do not apply.
