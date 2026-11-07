# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-11-07

First release of GRF-Net as a standalone package.

### Added

- `GRFNet`: learnable spatial front-end, frozen and learnable dilated convolution bank, gated fusion pooling and a configurable head.
- Inputs of any length; optional `n_times` and `sfreq`; explicit `dilations`; `scale_dilations` with half-up rounding.
- Geometry-seeded spatial filters from sensor coordinates in any number of dimensions, with duplicate and constant patterns skipped.
- `channel_positions` for 2-D and 3-D EEG electrode positions, in the optional `eeg` extra.
- `forward_features`, `n_outputs` with `n_classes` as an alias, headless models and custom heads.
- Pooling statistics `ppv`, `logvar` (alias `bandpower`), `mean`, `max`, `std` and custom functions; configurable `logvar_eps`.
- `bank_seed` for frozen kernels that do not depend on the global random number generator.
- `get_config`, `from_config`, `save`, `load` and `load_checkpoint`.
- Support for TorchScript, `torch.export`, `torch.compile` and mixed precision; `export_onnx` in the optional `onnx` extra.
- Reference-output tests that lock the default model's behaviour.
- Continuous integration on Python 3.11, 3.12 and 3.13, Ruff linting, `py.typed`, `CITATION.cff`.

### Changed

- Package layout: each model lives in `grfnet/models/<name>/` with its own README; shared layers are in `grfnet/layers.py` and checkpoint code in `grfnet/checkpoint.py`.
