# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `grfnet.data`: EEGMMIDB loader with six motor tasks, an optional rest class, resampling, band-pass filtering, channel selection and per-subject caching.
- `python -m grfnet.data report`: dataset figures (montage, class balance, amplitude, example trials, spectra, ERD/ERS, topomaps), `summary.json`, `recordings.csv` and `report.md`.
- `viz` extra with Matplotlib. Run outputs go under the ignored `runs/` folder.
- `python -m grfnet.train`: TOML-configured training with subject-disjoint folds, validation-based model selection, seeds, checkpoints, CSV and JSON logs, predictions and per-subject metrics.
- `python -m grfnet.evaluate`: checkpoint testing and a paired Wilcoxon comparison of two runs, in the optional `eval` extra.
- `grfnet.experiments`: config, split, metric and runner functions behind both commands.
- tqdm progress bars for training epochs (with a count of completed folds), dataset loading and the dataset survey, with `--no-progress` on both commands. tqdm is added to the `eeg` extra.
- `python -m grfnet.report`: run reports with training curves, a confusion matrix, per-subject and per-fold scores, spatial filters and activation patterns, kernel frequency responses, stream gates, a t-SNE of the embeddings, input saliency and an efficiency benchmark (parameters, operations, latency, throughput, training time), written as figures, `report.md`, `report.json` and `efficiency.json`.
- Standard run names, `{model}_{dataset}_{task}[_rest][_{variant}]`, derived from the config. Dataset reports go to `runs/dataset_{dataset}_{task}` and config files are named after their run.
- `data.normalize = "euclidean"`: per-subject Euclidean alignment (He and Wu, 2019), applied wherever runs load data, and `grfnet.data.euclidean_alignment`.
- Stopping and resuming training: Ctrl+C finishes the current fold and exits with status 130, a second Ctrl+C aborts, and re-running the same command resumes from the finished folds.
- scikit-learn in the `viz` extra, for t-SNE. Without it, embeddings are projected with PCA.
- `configs/grfnet_eegmmidb_imagery_lr.toml`: baseline with `dilations = [1, 2, 3, 5]` at 160 Hz.

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
