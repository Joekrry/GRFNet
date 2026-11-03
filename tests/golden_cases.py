"""Reference model configurations and outputs. Run this file to regenerate golden_outputs.json."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from grfnet import GRFNet

GOLDEN_PATH = Path(__file__).with_name("golden_outputs.json")
SEED = 1234
BATCH_SIZE = 4

_DIAMOND = [[0.0, 1.0], [1.0, 0.0], [0.0, -1.0], [-1.0, 0.0]]

CASES = {
    "default_22ch": {"n_channels": 22, "n_times": 500, "n_classes": 4},
    "low_rate_125hz": {"n_channels": 10, "n_times": 250, "n_classes": 4, "sfreq": 125.0},
    "high_rate_1000hz": {"n_channels": 3, "n_times": 500, "n_classes": 2, "sfreq": 1000.0},
    "geometry_coords": {"n_channels": 4, "n_times": 250, "n_classes": 2, "channel_coords": _DIAMOND},
    "fixed_bank_only": {"n_channels": 8, "n_times": 250, "n_classes": 3, "learnable_per_group": 0},
    "ppv_only": {"n_channels": 8, "n_times": 250, "n_classes": 3, "statistics": ["ppv"]},
    "no_ppv_bias": {"n_channels": 8, "n_times": 250, "n_classes": 3, "ppv_bias": False},
}


def compute_case(kwargs: dict, model_cls=GRFNet) -> dict:
    """Build a seeded model, then record its structure and its outputs before and after one training step."""
    torch.manual_seed(SEED)
    model = model_cls(**kwargs)

    gen = torch.Generator().manual_seed(SEED)
    x = torch.randn(BATCH_SIZE, kwargs["n_channels"], kwargs["n_times"], generator=gen)
    y = torch.randint(0, kwargs["n_classes"], (BATCH_SIZE,), generator=gen)

    model.eval()
    with torch.no_grad():
        logits_init = model(x)

    # Seeded separately to fix the dropout mask.
    torch.manual_seed(SEED + 1)
    model.train()
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=1e-3)
    optimizer.zero_grad()
    torch.nn.functional.cross_entropy(model(x), y).backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        logits_trained = model(x)

    return {
        "n_features": model.n_features,
        "dilations": list(model.bank.dilations),
        "trainable_parameters": model.count_parameters(),
        "state_dict_shapes": {name: list(value.shape) for name, value in model.state_dict().items()},
        "logits_init": logits_init.tolist(),
        "logits_trained": logits_trained.tolist(),
    }


if __name__ == "__main__":
    golden = {name: compute_case(kwargs) for name, kwargs in CASES.items()}
    GOLDEN_PATH.write_text(json.dumps(golden, indent=1) + "\n", encoding="utf-8")
