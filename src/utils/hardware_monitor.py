"""Observed resource monitoring for the Python client process.

The monitor does not estimate electrical energy, GPU use or per-query Ollama
cost. Ollama is a separate process and is reported as un-attributed.
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any

import psutil


class HardwareMonitor:
    schema_version = 2

    def __init__(self) -> None:
        self._process = psutil.Process()
        self._start_time: float | None = None
        self._snapshots: list[dict[str, Any]] = []

    def start(self, *, run_id: str | None = None, mode: str | None = None) -> None:
        self.run_id, self.mode = run_id, mode
        self._start_time = time.perf_counter()
        self._snapshots = []
        self._read(self._process.cpu_percent)  # prime psutil's interval-less CPU measurement

    def snapshot(self, stage: str = "unspecified") -> dict[str, Any]:
        elapsed = (time.perf_counter() - self._start_time) if self._start_time is not None else None
        cpu, cpu_error = self._read(self._process.cpu_percent)
        rss, rss_error = self._read(lambda: self._process.memory_info().rss / (1024 * 1024))
        threads, threads_error = self._read(self._process.num_threads)
        system_cpu, system_cpu_error = self._read(lambda: psutil.cpu_percent(interval=None))
        system_ram, system_ram_error = self._read(
            lambda: psutil.virtual_memory().used / (1024 * 1024)
        )
        return {
            "schema_version": self.schema_version,
            "run_id": getattr(self, "run_id", None),
            "mode": getattr(self, "mode", None),
            "stage": stage,
            "elapsed_s": elapsed,
            "process": {
                "scope": "python_process",
                "pid": self._process.pid,
                "cpu_percent": cpu,
                "rss_mb": rss,
                "num_threads": threads,
                "errors": {
                    k: v
                    for k, v in {
                        "cpu_percent": cpu_error,
                        "rss_mb": rss_error,
                        "num_threads": threads_error,
                    }.items()
                    if v
                },
            },
            "system": {
                "scope": "system",
                "cpu_percent": system_cpu,
                "ram_used_mb": system_ram,
                "errors": {
                    k: v
                    for k, v in {
                        "cpu_percent": system_cpu_error,
                        "ram_used_mb": system_ram_error,
                    }.items()
                    if v
                },
            },
            "ollama_process": {
                "status": "not_collected",
                "reason": "separate_process_not_attributed",
            },
            "energy_joules": None,
            "gpu": None,
        }

    def sample(self, stage: str) -> dict[str, Any]:
        snapshot = self.snapshot(stage)
        self._snapshots.append(snapshot)
        return snapshot

    def summarize(self) -> dict[str, Any]:
        snapshots = self._snapshots or [self.snapshot()]
        cpu = [
            s["process"]["cpu_percent"]
            for s in snapshots
            if s["process"]["cpu_percent"] is not None
        ]
        rss = [s["process"]["rss_mb"] for s in snapshots if s["process"]["rss_mb"] is not None]
        threads = [
            s["process"]["num_threads"]
            for s in snapshots
            if s["process"]["num_threads"] is not None
        ]
        return {
            "scope": "observed_python_process",
            "n_samples": len(snapshots),
            "cpu_percent_mean": statistics.fmean(cpu) if cpu else None,
            "cpu_percent_peak_observed": max(cpu) if cpu else None,
            "rss_mb_peak_observed": max(rss) if rss else None,
            "num_threads_peak_observed": max(threads) if threads else None,
            "total_elapsed_s": time.perf_counter() - self._start_time
            if self._start_time is not None
            else None,
            "energy_joules": None,
            "gpu": None,
        }

    def format_report(self) -> str:
        summary = self.summarize()
        return (
            "  === Recursos observados: processo Python ===\n"
            f"  Amostras: {summary['n_samples']}; CPU média/pico observado: "
            f"{summary['cpu_percent_mean']}/{summary['cpu_percent_peak_observed']}%\n"
            f"  RSS pico observado: {summary['rss_mb_peak_observed']} MB; "
            f"threads pico: {summary['num_threads_peak_observed']}\n"
            "  Energia/GPU: não coletadas; Ollama não é atribuído ao processo cliente.\n"
        )

    def save_json(self, path: str) -> None:
        Path(path).write_text(
            json.dumps({"summary": self.summarize(), "snapshots": self._snapshots}, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _read(callback: Any) -> tuple[float | int | None, str | None]:
        try:
            return callback(), None
        except (psutil.Error, OSError, ValueError, AttributeError) as error:
            return None, type(error).__name__
