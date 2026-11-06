from __future__ import annotations

import json
import sys

import numpy as np
import pytest

from conftest import requires_onnx, requires_torch

CONFIGS = [
    {"n_classes": 4},
    {"head": None},
    {"n_classes": 2, "learnable_per_group": 0, "statistics": ("max", "std")},
    {"n_outputs": 3, "ppv_bias": False, "dilations": [1, 3, 9]},
]


def _model(**kwargs):
    import torch

    from grfnet import GRFNet

    torch.manual_seed(0)
    return GRFNet(6, **kwargs).eval()


@requires_torch()
@pytest.mark.parametrize("kwargs", CONFIGS)
def test_torchscript_matches_eager(kwargs):
    import torch

    model = _model(**kwargs)
    scripted = torch.jit.script(model)
    for shape in [(2, 6, 200), (5, 6, 77)]:
        x = torch.randn(shape)
        with torch.no_grad():
            assert torch.allclose(scripted(x), model(x), atol=1e-6)


@requires_torch()
def test_torchscript_keeps_input_checks():
    import torch

    scripted = torch.jit.script(_model(n_classes=2))
    with pytest.raises(Exception, match="expected 6 channels"):
        scripted(torch.randn(2, 5, 50))


@requires_torch()
def test_torchscript_rejects_custom_statistics():
    import torch

    model = _model(n_classes=2, statistics=("mean", lambda feats: feats.amin(dim=-1)))
    with pytest.raises(ValueError, match="cannot be compiled with TorchScript"):
        torch.jit.script(model)


@requires_torch()
@pytest.mark.parametrize("kwargs", CONFIGS)
def test_torch_export_with_dynamic_shapes(kwargs):
    import torch

    model = _model(**kwargs)
    dims = {"x": {0: torch.export.Dim("batch", min=1), 2: torch.export.Dim("n_times", min=2)}}
    program = torch.export.export(model, (torch.randn(2, 6, 200),), dynamic_shapes=dims)
    for shape in [(1, 6, 200), (4, 6, 333)]:
        x = torch.randn(shape)
        with torch.no_grad():
            assert torch.allclose(program.module()(x), model(x), atol=1e-6)


@requires_torch()
def test_torch_compile_full_graph():
    import torch

    model = _model(n_classes=4)
    compiled = torch.compile(model, backend="aot_eager", fullgraph=True)
    x = torch.randn(3, 6, 128)
    with torch.no_grad():
        assert torch.allclose(compiled(x), model(x), atol=1e-6)


@requires_torch()
def test_bfloat16_autocast_on_cpu():
    import torch

    model = _model(n_classes=4).train()
    x = torch.randn(4, 6, 128)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = model(x)
    assert torch.isfinite(out).all()
    out.float().sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


@requires_torch()
def test_float16_autocast_on_cuda():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    model = _model(n_classes=4).cuda().train()
    x = torch.randn(4, 6, 128, device="cuda")
    scaler = torch.amp.GradScaler("cuda")
    with torch.autocast("cuda", dtype=torch.float16):
        out = model(x)
        loss = torch.nn.functional.cross_entropy(out, torch.randint(0, 4, (4,), device="cuda"))
    assert torch.isfinite(out).all()
    scaler.scale(loss).backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


@requires_torch()
@pytest.mark.parametrize("kwargs", CONFIGS + [{"n_classes": 2, "channel_coords": np.random.default_rng(0).standard_normal((6, 3))}])
def test_config_round_trip(kwargs):
    from grfnet import GRFNet

    model = _model(**kwargs)
    config = model.get_config()
    json.dumps(config)
    rebuilt = GRFNet.from_config(json.loads(json.dumps(config)))
    assert rebuilt.get_config() == config
    assert {k: v.shape for k, v in rebuilt.state_dict().items()} == {k: v.shape for k, v in model.state_dict().items()}


@requires_torch()
def test_config_keeps_original_arguments():
    model = _model(n_classes=3, sfreq=125.0)
    config = model.get_config()
    assert config["sfreq"] == 125.0
    assert config["dilations"] is None
    assert config["statistics"] == ["ppv", "logvar", "mean"]


@requires_torch()
def test_config_rejects_custom_parts():
    from torch import nn

    with pytest.raises(ValueError, match="custom head"):
        _model(head=nn.Linear(576, 2)).get_config()
    with pytest.raises(ValueError, match="custom statistics"):
        _model(n_classes=2, statistics=(lambda feats: feats.mean(dim=-1),)).get_config()


@requires_torch()
@pytest.mark.parametrize("kwargs", CONFIGS)
def test_save_and_load_reproduce_outputs(tmp_path, kwargs):
    import torch

    from grfnet import GRFNet

    model = _model(**kwargs)
    path = tmp_path / "model.pt"
    model.save(path, metadata={"epoch": 12, "note": "test"})
    loaded = GRFNet.load(path)
    assert not loaded.training
    x = torch.randn(3, 6, 150)
    with torch.no_grad():
        assert torch.equal(loaded(x), model(x))


@requires_torch()
def test_checkpoint_contents(tmp_path):
    from grfnet import __version__, load_checkpoint

    model = _model(n_classes=2)
    path = tmp_path / "model.pt"
    model.save(path, metadata={"epoch": 3})
    checkpoint = load_checkpoint(path)
    assert checkpoint["model"] == "GRFNet"
    assert checkpoint["grfnet_version"] == __version__
    assert checkpoint["metadata"] == {"epoch": 3}
    assert checkpoint["config"] == model.get_config()


@requires_torch()
def test_load_rejects_other_files(tmp_path):
    import torch

    from grfnet import GRFNet

    path = tmp_path / "other.pt"
    torch.save({"weights": torch.zeros(3)}, path)
    with pytest.raises(ValueError, match="not a GRFNet checkpoint"):
        GRFNet.load(path)


@requires_torch()
def test_load_rejects_other_models(tmp_path):
    import torch

    from grfnet import GRFNet, load_checkpoint

    path = tmp_path / "model.pt"
    _model(n_classes=2).save(path)
    checkpoint = load_checkpoint(path)
    checkpoint["model"] = "OtherNet"
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="holds a OtherNet"):
        GRFNet.load(path)


@requires_onnx()
@requires_torch()
@pytest.mark.parametrize("kwargs", CONFIGS)
def test_onnx_export_matches_eager(tmp_path, kwargs):
    import onnxruntime
    import torch

    from grfnet import export_onnx

    model = _model(**kwargs)
    path = tmp_path / "model.onnx"
    export_onnx(model, path, n_times=200)
    session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    for shape in [(2, 6, 200), (1, 6, 77), (5, 6, 512)]:
        x = torch.randn(shape)
        out = session.run(None, {"input": x.numpy()})[0]
        with torch.no_grad():
            np.testing.assert_allclose(out, model(x).numpy(), atol=1e-5)


def test_onnx_export_explains_missing_dependencies(monkeypatch, tmp_path):
    from grfnet import export_onnx

    monkeypatch.setitem(sys.modules, "onnx", None)
    with pytest.raises(ImportError, match=r"grfnet\[onnx\]"):
        export_onnx(object(), tmp_path / "model.onnx")
