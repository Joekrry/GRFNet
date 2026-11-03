from __future__ import annotations

import pytest

from conftest import requires_torch


@requires_torch()
def test_n_times_is_optional():
    import torch

    from grfnet import GRFNet

    model = GRFNet(n_channels=4, n_classes=3)
    assert model.n_times is None
    assert model(torch.randn(2, 4, 300)).shape == (2, 3)


@requires_torch()
@pytest.mark.parametrize("n_times", [1, 17, 300, 2048])
def test_accepts_inputs_of_any_length(n_times):
    import torch

    from grfnet import GRFNet

    model = GRFNet(4, 1000, 2).eval()
    with torch.no_grad():
        out = model(torch.randn(2, 4, n_times))
    assert out.shape == (2, 2)
    assert torch.isfinite(out).all()


@requires_torch()
def test_positional_arguments_still_work():
    from grfnet import GRFNet

    model = GRFNet(22, 1000, 4)
    assert (model.n_channels, model.n_times, model.n_classes) == (22, 1000, 4)


@requires_torch()
def test_requires_n_classes():
    from grfnet import GRFNet

    with pytest.raises(TypeError, match="n_classes"):
        GRFNet(n_channels=4, n_times=100)


@requires_torch()
@pytest.mark.parametrize("kwargs, match", [
    ({"n_times": 0}, "n_times must be >= 1"),
    ({"sfreq": 0.0}, "sfreq must be positive"),
    ({"sfreq": -250.0}, "sfreq must be positive"),
])
def test_rejects_invalid_sizes_and_rates(kwargs, match):
    from grfnet import GRFNet

    with pytest.raises(ValueError, match=match):
        GRFNet(n_channels=4, n_classes=2, **kwargs)


@requires_torch()
def test_without_sfreq_uses_base_dilations_unscaled():
    from grfnet import GRFNet

    assert GRFNet(4, n_classes=2).bank.dilations == [1, 2, 4, 8]
    assert GRFNet(4, n_classes=2, sfreq=250.0).bank.dilations == [1, 2, 4, 8]
    assert GRFNet(4, n_classes=2, base_dilations=(3, 5)).bank.dilations == [3, 5]


@requires_torch()
def test_explicit_dilations_are_used_as_given():
    from grfnet import GRFNet

    model = GRFNet(4, n_classes=2, dilations=[1, 3, 9, 27, 81])
    assert model.bank.dilations == [1, 3, 9, 27, 81]
    assert len(model.bank.fixed) == 5
    assert model.n_features == 5 * 16 * 3


@requires_torch()
@pytest.mark.parametrize("kwargs", [{"sfreq": 125.0}, {"base_dilations": (1, 2)}])
def test_explicit_dilations_conflict_with_scaling_options(kwargs):
    from grfnet import GRFNet

    with pytest.raises(ValueError, match="not both"):
        GRFNet(4, n_classes=2, dilations=[1, 2], **kwargs)


@requires_torch()
@pytest.mark.parametrize("dilations", [[], [0, 1], [1, 2.5], (-1,)])
def test_rejects_invalid_dilations(dilations):
    from grfnet import GRFNet

    with pytest.raises(ValueError, match="dilations must be"):
        GRFNet(4, n_classes=2, dilations=dilations)


@requires_torch()
def test_unscaled_base_dilations_must_be_integers():
    from grfnet import GRFNet

    with pytest.raises(ValueError, match="dilations must be integers"):
        GRFNet(4, n_classes=2, base_dilations=(1.5, 3))


@requires_torch()
@pytest.mark.parametrize("sfreq, expected", [
    (100.0, [1, 1, 2, 3]),
    (125.0, [1, 1, 2, 4]),
    (375.0, [2, 3, 6, 12]),
    (625.0, [3, 5, 10, 20]),
    (1000.0, [4, 8, 16, 32]),
])
def test_scale_dilations_rounds_half_up(sfreq, expected):
    from grfnet import scale_dilations

    assert scale_dilations((1, 2, 4, 8), sfreq) == expected


def test_scale_dilations_custom_reference_rate():
    from grfnet import scale_dilations

    assert scale_dilations((1, 2, 4), 16000.0, reference_sfreq=8000.0) == [2, 4, 8]
    with pytest.raises(ValueError, match="must be positive"):
        scale_dilations((1,), 100.0, reference_sfreq=0.0)
