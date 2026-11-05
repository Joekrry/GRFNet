from __future__ import annotations

import pytest

from conftest import requires_torch


def _seeded(seed=0, **kwargs):
    import torch

    from grfnet import GRFNet

    torch.manual_seed(seed)
    return GRFNet(**kwargs)


@requires_torch()
def test_n_outputs_matches_n_classes():
    import torch

    by_classes = _seeded(n_channels=6, n_classes=3).eval()
    by_outputs = _seeded(n_channels=6, n_outputs=3).eval()
    assert by_outputs.n_classes == by_outputs.n_outputs == 3
    x = torch.randn(2, 6, 128)
    with torch.no_grad():
        assert torch.equal(by_classes(x), by_outputs(x))


@requires_torch()
def test_n_classes_and_n_outputs_must_agree():
    from grfnet import GRFNet

    assert GRFNet(6, n_classes=3, n_outputs=3).n_outputs == 3
    with pytest.raises(ValueError, match="disagree"):
        GRFNet(6, n_classes=3, n_outputs=4)


@requires_torch()
def test_forward_features_feed_the_head():
    import torch

    model = _seeded(n_channels=6, n_classes=3).eval()
    x = torch.randn(4, 6, 128)
    with torch.no_grad():
        embedding = model.forward_features(x)
        assert embedding.shape == (4, model.embedding_dim)
        assert torch.allclose(model.classify(embedding), model(x))


@requires_torch()
def test_headless_model_returns_embeddings():
    import torch

    from grfnet import GRFNet

    model = GRFNet(6, head=None).eval()
    assert model.classify is None
    assert model.n_outputs is None
    assert not any(name.startswith("classify.") for name in model.state_dict())
    with torch.no_grad():
        out = model(torch.randn(3, 6, 100))
    assert out.shape == (3, model.embedding_dim)


@requires_torch()
def test_headless_model_rejects_output_size():
    from grfnet import GRFNet

    with pytest.raises(ValueError, match="leave n_outputs"):
        GRFNet(6, n_outputs=3, head=None)


@requires_torch()
def test_linear_head_needs_output_size():
    from grfnet import GRFNet

    with pytest.raises(TypeError, match="n_outputs or n_classes"):
        GRFNet(6)


@requires_torch()
def test_custom_head_module():
    import torch
    from torch import nn

    from grfnet import GRFNet

    probe = GRFNet(6, head=None)
    head = nn.Sequential(nn.Linear(probe.embedding_dim, 16), nn.ReLU(), nn.Linear(16, 5))
    model = GRFNet(6, head=head)
    assert model.classify is head
    assert model(torch.randn(2, 6, 64)).shape == (2, 5)


@requires_torch()
def test_custom_head_factory_receives_embedding_dim():
    import torch
    from torch import nn

    from grfnet import GRFNet

    seen = []

    def factory(dim):
        seen.append(dim)
        return nn.Linear(dim, 2)

    model = GRFNet(6, head=factory)
    assert seen == [model.embedding_dim]
    assert model(torch.randn(2, 6, 64)).shape == (2, 2)


@requires_torch()
@pytest.mark.parametrize("head", ["mlp", 3])
def test_rejects_unknown_head(head):
    from grfnet import GRFNet

    with pytest.raises(ValueError, match="head must be"):
        GRFNet(6, n_outputs=2, head=head)


@requires_torch()
def test_head_factory_must_return_module():
    from grfnet import GRFNet

    with pytest.raises(ValueError, match="must return an nn.Module"):
        GRFNet(6, head=lambda dim: "not a module")


@requires_torch()
def test_regression_and_multilabel_train():
    import torch

    from grfnet import GRFNet

    x = torch.randn(8, 6, 96)
    regressor = GRFNet(6, n_outputs=1)
    torch.nn.functional.mse_loss(regressor(x).squeeze(1), torch.randn(8)).backward()
    assert regressor.classify.weight.grad is not None

    tagger = GRFNet(6, n_outputs=4)
    targets = torch.randint(0, 2, (8, 4)).float()
    torch.nn.functional.binary_cross_entropy_with_logits(tagger(x), targets).backward()
    assert tagger.classify.weight.grad is not None


@requires_torch()
def test_bandpower_is_an_alias_of_logvar():
    import torch

    x = torch.randn(2, 6, 128)
    a = _seeded(n_channels=6, n_classes=3, statistics=("ppv", "logvar", "mean")).eval()
    b = _seeded(n_channels=6, n_classes=3, statistics=("ppv", "bandpower", "mean")).eval()
    with torch.no_grad():
        assert torch.equal(a(x), b(x))


@requires_torch()
@pytest.mark.parametrize("statistics", [("logvar", "bandpower"), ("mean", "mean"), ()])
def test_rejects_repeated_or_empty_statistics(statistics):
    from grfnet import GRFNet

    with pytest.raises(ValueError, match="statistics must"):
        GRFNet(6, n_classes=2, statistics=statistics)


@requires_torch()
@pytest.mark.parametrize("stat", ["median", 3])
def test_rejects_unknown_statistic(stat):
    from grfnet import GRFNet

    with pytest.raises(ValueError, match="statistics must be callables or names"):
        GRFNet(6, n_classes=2, statistics=("mean", stat))


@requires_torch()
def test_max_and_std_statistics():
    import torch

    model = _seeded(n_channels=6, n_classes=2, statistics=("max", "std")).eval()
    assert model.embedding_dim == 2 * model.n_features
    assert model.ppv_bias is None
    with torch.no_grad():
        assert torch.isfinite(model(torch.randn(2, 6, 64))).all()


@requires_torch()
def test_std_gradient_is_finite_on_constant_input():
    import torch

    model = _seeded(n_channels=6, n_classes=2, statistics=("std",))
    model(torch.zeros(2, 6, 64)).sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


@requires_torch()
def test_custom_statistic_callable():
    import torch

    def peak_to_peak(feats):
        return feats.amax(dim=-1) - feats.amin(dim=-1)

    model = _seeded(n_channels=6, n_classes=2, statistics=("ppv", peak_to_peak)).eval()
    assert model.embedding_dim == 2 * model.n_features
    with torch.no_grad():
        assert model(torch.randn(2, 6, 64)).shape == (2, 2)


@requires_torch()
def test_custom_statistic_with_wrong_shape_raises():
    import torch

    model = _seeded(n_channels=6, n_classes=2, statistics=(lambda feats: feats,))
    with pytest.raises(ValueError, match="custom statistic must return"):
        model(torch.randn(2, 6, 64))


@requires_torch()
def test_logvar_eps_is_configurable():
    import torch

    x = torch.zeros(2, 6, 64)
    a = _seeded(n_channels=6, n_classes=2, statistics=("logvar",)).eval()
    b = _seeded(n_channels=6, n_classes=2, statistics=("logvar",), logvar_eps=1e-2).eval()
    with torch.no_grad():
        assert not torch.allclose(a(x), b(x))
    with pytest.raises(ValueError, match="logvar_eps must be positive"):
        _seeded(n_channels=6, n_classes=2, logvar_eps=0.0)


def _fixed_kernels(model):
    import torch

    return torch.cat([conv.weight.flatten() for conv in model.bank.fixed])


@requires_torch()
def test_bank_seed_fixes_frozen_kernels_independently_of_global_rng():
    import torch

    a = _seeded(seed=1, n_channels=6, n_classes=2, bank_seed=7)
    b = _seeded(seed=2, n_channels=6, n_classes=2, bank_seed=7)
    c = _seeded(seed=1, n_channels=6, n_classes=2, bank_seed=8)
    assert torch.equal(_fixed_kernels(a), _fixed_kernels(b))
    assert not torch.equal(_fixed_kernels(a), _fixed_kernels(c))
    assert all(not conv.weight.requires_grad for conv in a.bank.fixed)


@requires_torch()
def test_bank_seed_kernels_stay_mean_centred():
    import torch

    model = _seeded(n_channels=6, n_classes=2, bank_seed=3)
    for conv in model.bank.fixed:
        assert torch.allclose(conv.weight.mean(dim=-1), torch.zeros(()), atol=1e-6)


@requires_torch()
def test_without_bank_seed_kernels_follow_global_rng():
    import torch

    a = _seeded(seed=5, n_channels=6, n_classes=2)
    b = _seeded(seed=5, n_channels=6, n_classes=2)
    c = _seeded(seed=6, n_channels=6, n_classes=2)
    assert torch.equal(_fixed_kernels(a), _fixed_kernels(b))
    assert not torch.equal(_fixed_kernels(a), _fixed_kernels(c))
