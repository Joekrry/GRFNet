# GRF-Net: Geometry-aware Rocket-Fusion Network

[![Tests](https://github.com/Joekrry/GRFNet/actions/workflows/tests.yml/badge.svg)](https://github.com/Joekrry/GRFNet/actions/workflows/tests.yml)
[![Licence: MIT](https://img.shields.io/badge/Licence-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.2+](https://img.shields.io/badge/PyTorch-2.2%2B-ee4c2c.svg)](https://pytorch.org/)

Yes I AI generate my personal project README's

Compact PyTorch models for multichannel time series. GRF-Net models combine
learnable spatial filters with a mostly frozen ROCKET-style convolution bank,
then fuse several temporal statistics into an embedding for a small trainable
head.

## Models

| Model                                       | Mechanism                                                                 | State     | Output                   |
| ------------------------------------------- | ------------------------------------------------------------------------- | --------- | ------------------------ |
| [GRFNet](src/grfnet/models/grfnet/README.md) | Spatial filters, frozen and learnable dilated kernels, gated fusion pooling | Stateless | One vector per sequence |

Each model's README covers its architecture, parameters, configuration,
limitations and references.

## Installation

```bash
git clone https://github.com/Joekrry/GRFNet.git
```

```bash
cd GRFNet
```

```bash
pip install -e .
```

Requires Python 3.11 or later and PyTorch 2.2 or later. Optional extras add
features:

| Extra  | Adds                             | Needed for                           |
| ------ | -------------------------------- | ------------------------------------ |
| `eeg`  | MNE-Python                       | Looking up EEG electrode positions   |
| `onnx` | onnx, onnxscript, ONNX Runtime   | `export_onnx`                        |
| `dev`  | pytest, Ruff                     | Running the tests and linter         |

```bash
pip install -e ".[eeg,onnx]"
```

## Quick start

```python
import torch
from grfnet import GRFNet

model = GRFNet(n_channels=22, n_times=1000, n_classes=4, sfreq=250.0)

x = torch.randn(8, 22, 1000)  # (batch, n_channels, n_times)
logits = model(x)             # (batch, n_classes)
```

The model returns raw scores for use with `torch.nn.CrossEntropyLoss`. See the
[GRFNet README](src/grfnet/models/grfnet/README.md) for regression, embeddings,
custom heads, pooling statistics and geometry-seeded spatial filters.

## Saving and deployment

### Checkpoints

`save` writes the model's configuration, weights and optional metadata to one
file. `load` rebuilds the model from it and returns it in eval mode:

```python
from grfnet import GRFNet, load_checkpoint

model = GRFNet(n_channels=8, n_classes=2)
model.save("grfnet.pt", metadata={"epoch": 40})

restored = GRFNet.load("grfnet.pt")
print(load_checkpoint("grfnet.pt")["metadata"])  # {'epoch': 40}
```

Checkpoints are read with `torch.load(..., weights_only=True)`. `get_config`
returns the constructor arguments as JSON-compatible values, and `from_config`
builds an untrained model from them. Custom heads and custom statistics are
Python objects and cannot be stored: for those models, rebuild the model in code
and load its `state_dict`.

### Export and compilation

| Target          | How                                                     | Notes                                                                                                  |
| --------------- | ------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| `torch.export`  | `torch.export.export(model, (x,), dynamic_shapes=...)` | Batch size and input length can be dynamic.                                                            |
| ONNX            | `export_onnx(model, "grfnet.onnx")`                     | Batch size and input length are dynamic by default. Needs the `onnx` extra.                            |
| TorchScript     | `torch.jit.script(model)`                               | Not available with custom statistics. PyTorch no longer supports TorchScript on Python 3.14 or later. |
| `torch.compile` | `torch.compile(model)`                                  | Compiles as a single graph.                                                                            |

```python
import torch
from grfnet import GRFNet, export_onnx

model = GRFNet(n_channels=8, n_classes=2).eval()
export_onnx(model, "grfnet.onnx", n_times=256)

batch, n_times = torch.export.Dim("batch", min=1), torch.export.Dim("n_times", min=2)
program = torch.export.export(model, (torch.randn(2, 8, 256),), dynamic_shapes={"x": {0: batch, 2: n_times}})
```

The ONNX file has one input, `input`, of shape `(batch, n_channels, n_times)`,
and one output, `output`.

### Mixed precision

The models run under `torch.autocast`, with `float16` on CUDA and `bfloat16` on
CPU.

## Repository structure

```text
src/grfnet/
  __init__.py        public API
  layers.py          shared spatial front-end, convolution bank and dilation scaling
  checkpoint.py      checkpoint files
  export.py          ONNX export
  channels.py        EEG electrode positions (eeg extra)
  models/
    grfnet/          GRFNet model and its README
tests/               unit tests and reference outputs
```

## Testing

```bash
pip install -e ".[dev]"
```

```bash
python -m pytest
```

Tests for optional features are skipped when their extra is not installed. To
run every test, install all extras:

```bash
pip install -e ".[dev,eeg,onnx]"
```

To check the code with the linter:

```bash
ruff check .
```

## Citation

If you use GRF-Net in your work, please cite this repository. Citation details
are also in [`CITATION.cff`](CITATION.cff).

```bibtex
@software{kerry2026grfnet,
  author  = {Kerry, Joe},
  title   = {{GRF-Net: Geometry-aware Rocket-Fusion Network}},
  version = {0.1.0},
  year    = {2026},
  url     = {https://github.com/Joekrry/GRFNet}
}
```

## Changes

See [`CHANGELOG.md`](CHANGELOG.md).

## Licence

Released under the [MIT License](LICENSE).
