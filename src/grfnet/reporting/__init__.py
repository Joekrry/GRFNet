"""Figures, tables and efficiency benchmarks for trained runs."""

from .benchmark import benchmark_model
from .report import read_run, write_run_report

__all__ = ["benchmark_model", "read_run", "write_run_report"]
