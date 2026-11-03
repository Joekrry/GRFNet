"""Locks the model's default behaviour against reference outputs in golden_outputs.json."""

from __future__ import annotations

import json

import numpy as np
import pytest

from conftest import requires_torch
from golden_cases import CASES, GOLDEN_PATH

# Absorbs floating-point differences between platforms.
TOLERANCE = 1e-4


@pytest.fixture(scope="module")
def golden():
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module", params=sorted(CASES))
def case(request, golden):
    from golden_cases import compute_case

    return compute_case(CASES[request.param]), golden[request.param]


def test_golden_file_covers_every_case(golden):
    assert set(golden) == set(CASES)


@requires_torch()
def test_structure_matches(case):
    actual, expected = case
    assert actual["n_features"] == expected["n_features"]
    assert actual["dilations"] == expected["dilations"]
    assert actual["trainable_parameters"] == expected["trainable_parameters"]


@requires_torch()
def test_state_dict_layout_matches(case):
    actual, expected = case
    assert actual["state_dict_shapes"] == expected["state_dict_shapes"]


@requires_torch()
def test_initial_outputs_match(case):
    actual, expected = case
    np.testing.assert_allclose(actual["logits_init"], expected["logits_init"], atol=TOLERANCE)


@requires_torch()
def test_outputs_after_training_step_match(case):
    actual, expected = case
    np.testing.assert_allclose(actual["logits_trained"], expected["logits_trained"], atol=TOLERANCE)
