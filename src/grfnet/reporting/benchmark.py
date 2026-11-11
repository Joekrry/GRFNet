"""Parameter counts, FLOPs, latency, throughput and training time."""

from __future__ import annotations

import logging
import platform
import statistics
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch


def count_parameters(model: torch.nn.Module) -> dict:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return {"trainable": int(trainable), "frozen": int(frozen), "total": int(trainable + frozen)}


@torch.no_grad()
def count_flops(model: torch.nn.Module, n_channels: int, n_times: int) -> int | None:
    """FLOPs for one trial. torch's counter covers convolutions and matmuls only."""
    # Triton is unused here; silence its import warning.
    logging.getLogger("torch.utils.flop_counter").setLevel(logging.ERROR)
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except ImportError:
        return None
    model.eval().cpu()
    counter = FlopCounterMode(display=False)
    with counter:
        model(torch.zeros(1, n_channels, n_times))
    return int(counter.get_total_flops())


@torch.inference_mode()
def time_inference(model: torch.nn.Module, n_channels: int, n_times: int, *, batch_size: int, device: str,
                   repeats: int = 30, warmup: int = 5) -> dict:
    model.eval().to(device)
    x = torch.randn(batch_size, n_channels, n_times, device=device)
    cuda = torch.device(device).type == "cuda"
    times = []
    for i in range(warmup + repeats):
        if cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        model(x)
        if cuda:
            torch.cuda.synchronize()
        if i >= warmup:
            times.append(time.perf_counter() - start)
    median = statistics.median(times)
    return {"device": device, "batch_size": batch_size, "repeats": repeats, "median_ms": 1000.0 * median,
            "trials_per_second": batch_size / median}


def training_time(histories: Mapping[int, Sequence[dict]]) -> dict:
    # The first epoch includes warm-up and is left out of the median.
    epoch_times = [float(r["seconds"]) for rows in histories.values() for r in rows[1:]]
    fold_totals = {int(k): float(sum(float(r["seconds"]) for r in rows)) for k, rows in histories.items()}
    return {
        "median_epoch_seconds": statistics.median(epoch_times) if epoch_times else None,
        "fold_seconds": fold_totals,
        "total_seconds": float(sum(fold_totals.values())),
    }


def benchmark_model(model: torch.nn.Module, n_channels: int, n_times: int, *, checkpoint: str | Path | None = None,
              histories: Mapping[int, Sequence[dict]] | None = None, repeats: int = 30) -> dict:
    """Benchmark a trained model on this machine."""
    result = {
        "parameters": count_parameters(model),
        "flops_per_trial": count_flops(model, n_channels, n_times),
        "input_shape": [n_channels, n_times],
        "checkpoint_bytes": Path(checkpoint).stat().st_size if checkpoint else None,
        "machine": {"platform": platform.platform(), "processor": platform.processor(),
                    "torch": torch.__version__, "cpu_threads": torch.get_num_threads(),
                    "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
        "inference": [time_inference(model, n_channels, n_times, batch_size=1, device="cpu", repeats=repeats),
                      time_inference(model, n_channels, n_times, batch_size=64, device="cpu", repeats=repeats)],
    }
    if torch.cuda.is_available():
        result["inference"] += [
            time_inference(model, n_channels, n_times, batch_size=1, device="cuda", repeats=repeats),
            time_inference(model, n_channels, n_times, batch_size=256, device="cuda", repeats=repeats),
        ]
    model.cpu()
    if histories:
        result["training"] = training_time(histories)
    return result
