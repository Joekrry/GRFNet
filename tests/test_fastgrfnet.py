from __future__ import annotations

from typing import Any

import pytest
import torch

from grfnet import FastGRFNet, GRFNet

ARGS: dict[str, Any] = dict(n_channels=6, n_times=200, n_classes=3, dilations=[1, 2], spatial_components=4)


def _fast(**kwargs) -> FastGRFNet:
    torch.manual_seed(0)
    return FastGRFNet(**{**ARGS, **kwargs})


def _activate_writer(model: FastGRFNet) -> None:
    # The value projection starts at zero; give it weights so that writes are non-zero.
    assert model.writer is not None
    with torch.no_grad():
        model.writer.value.weight.normal_(0, 1.0)
        model.writer.value.bias.normal_(0, 1.0)


@pytest.mark.parametrize("mode", ["off", "unsupervised", "label_feedback"])
def test_zero_state_reproduces_grfnet(mode):
    torch.manual_seed(1)
    reference = GRFNet(**ARGS).eval()
    model = _fast(write_mode=mode).eval()
    missing, unexpected = model.load_state_dict(reference.state_dict(), strict=False)
    assert not unexpected and all(name.startswith("writer.") for name in missing)
    x = torch.randn(5, 6, 200)
    logits, state = model(x)
    torch.testing.assert_close(logits, reference(x), atol=1e-5, rtol=1e-5)
    assert state.shape == (5, 4, 6)


def test_new_writer_keeps_the_state_at_zero():
    model = _fast().eval()
    _, state = model(torch.randn(3, 6, 200))
    assert torch.count_nonzero(state) == 0


def test_off_never_writes():
    model = _fast(write_mode="off").eval()
    assert model.writer is None
    start = torch.randn(2, 4, 6)
    _, state = model(torch.randn(2, 6, 200), start)
    assert torch.equal(state, start)


def test_writes_change_later_trials_but_not_the_current_one():
    model = _fast().eval()
    _activate_writer(model)
    x1, x2 = torch.randn(2, 6, 200), torch.randn(2, 6, 200)
    first, state = model(x1)
    assert torch.count_nonzero(state) > 0
    torch.testing.assert_close(first, model(x1)[0])
    later, _ = model(x2, state)
    assert not torch.allclose(later, model(x2)[0])


def test_label_feedback_never_affects_the_current_trial():
    model = _fast(write_mode="label_feedback").eval()
    _activate_writer(model)
    x = torch.randn(4, 6, 200)
    a, state_a = model(x, label=torch.tensor([0, 1, 2, 0]))
    b, state_b = model(x, label=torch.tensor([2, 2, 2, 2]))
    torch.testing.assert_close(a, b)
    assert not torch.allclose(state_a, state_b)


def test_unknown_labels_equal_no_labels():
    model = _fast(write_mode="label_feedback").eval()
    _activate_writer(model)
    x = torch.randn(3, 6, 200)
    _, with_unknown = model(x, label=torch.full((3,), -1))
    _, without = model(x)
    torch.testing.assert_close(with_unknown, without)


def test_fast_weight_norm_is_clipped():
    model = _fast(eta_max=5.0, delta_clip=0.3).eval()
    _activate_writer(model)
    state = None
    for _ in range(20):
        _, state = model(torch.randn(2, 6, 200) * 10, state)
        assert state.flatten(1).norm(dim=1).max() <= 0.3 + 1e-5


def test_decay_shrinks_fast_weights_without_writes():
    model = _fast(eta_max=0.0, decay=0.5).eval()
    _activate_writer(model)
    start = torch.randn(2, 4, 6) * 0.1
    _, state = model(torch.randn(2, 6, 200), start)
    torch.testing.assert_close(state, 0.5 * start)


def test_gradients_reach_the_writer():
    model = _fast()
    _activate_writer(model)
    _, state = model(torch.randn(4, 6, 200))
    logits, _ = model(torch.randn(4, 6, 200), state)
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 2, 0])).backward()
    assert model.writer is not None
    for name, param in model.writer.named_parameters():
        assert param.grad is not None and param.grad.abs().sum() > 0, name


def test_any_length_and_input_checks():
    model = _fast().eval()
    assert model(torch.randn(2, 6, 333))[0].shape == (2, 3)
    with pytest.raises(ValueError, match="channels"):
        model(torch.randn(2, 5, 200))
    with pytest.raises(ValueError, match="state must have shape"):
        model(torch.randn(2, 6, 200), torch.zeros(3, 4, 6))
    with pytest.raises(ValueError, match="label must have shape"):
        _fast(write_mode="label_feedback")(torch.randn(2, 6, 200), label=torch.tensor([0]))


@pytest.mark.parametrize("bad", [dict(write_mode="always"), dict(decay=1.0), dict(delta_clip=0.0)])
def test_bad_arguments_are_rejected(bad):
    with pytest.raises(ValueError):
        _fast(**bad)


def test_label_feedback_needs_classes():
    with pytest.raises(ValueError, match="n_classes"):
        FastGRFNet(n_channels=6, head=None, write_mode="label_feedback")


def test_checkpoint_round_trip(tmp_path):
    model = _fast(write_mode="label_feedback", decay=0.8).eval()
    _activate_writer(model)
    model.save(tmp_path / "fast.pt")
    loaded = FastGRFNet.load(tmp_path / "fast.pt")
    assert loaded.get_config()["write_mode"] == "label_feedback" and loaded.decay == 0.8
    x, label = torch.randn(2, 6, 200), torch.tensor([1, 2])
    for a, b in zip(model(x, label=label), loaded(x, label=label), strict=True):
        torch.testing.assert_close(a, b)


def test_trainable_parameters_add_only_the_writer():
    plain = GRFNet(**ARGS).count_parameters()
    assert _fast(write_mode="off").count_parameters() == plain
    model = _fast()
    assert model.writer is not None
    assert model.count_parameters() == plain + sum(p.numel() for p in model.writer.parameters())
