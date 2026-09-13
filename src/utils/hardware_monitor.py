import json
import statistics
import time
from pathlib import Path
from typing import Any

import psutil


class HardwareMonitor:
    def __init__(self) -> None:
        self._process = psutil.Process()
        self._start_time: float | None = None
        self._snapshots: list[dict[str, Any]] = []

    def start(self) -> None:
        self._start_time = time.time()
        self._snapshots = []

    def snapshot(self) -> dict:
        now = time.time()
        elapsed = (now - self._start_time) if self._start_time is not None else 0.0

        cpu_percent = self._safe(self._process.cpu_percent)
        rss_mb = self._safe(lambda: self._process.memory_info().rss / (1024 * 1024))
        num_threads = self._safe(self._process.num_threads)

        return {
            "timestamp": now,
            "cpu_percent": cpu_percent,
            "rss_mb": rss_mb,
            "num_threads": num_threads,
            "elapsed_s": elapsed,
        }

    def summarize(self) -> dict:
        snapshots = self._snapshots
        if not snapshots:
            snapshots = [self.snapshot()]

        cpu_values = [s["cpu_percent"] for s in snapshots if s["cpu_percent"] is not None]
        rss_values = [s["rss_mb"] for s in snapshots if s["rss_mb"] is not None]
        thread_values = [s["num_threads"] for s in snapshots if s["num_threads"] is not None]

        return {
            "cpu_percent_mean": statistics.fmean(cpu_values) if cpu_values else 0.0,
            "cpu_percent_peak": max(cpu_values) if cpu_values else 0.0,
            "rss_mb_peak": max(rss_values) if rss_values else 0.0,
            "num_threads_peak": max(thread_values) if thread_values else 0,
            "total_elapsed_s": time.time() - self._start_time if self._start_time is not None else 0.0,
        }

    def format_report(self) -> str:
        summary = self.summarize()
        return (
            "  === Resumo de Hardware ===\n"
            f"  CPU média: {summary['cpu_percent_mean']:.1f}%\n"
            f"  CPU pico: {summary['cpu_percent_peak']:.1f}%\n"
            f"  RAM pico (RSS): {summary['rss_mb_peak']:.1f} MB\n"
            f"  Threads máximas: {summary['num_threads_peak']}\n"
            f"  Duração total: {summary['total_elapsed_s']:.1f}s\n"
        )

    def save_json(self, path: str) -> None:
        data = {
            "summary": self.summarize(),
            "snapshots": [
                {
                    "timestamp": s["timestamp"],
                    "cpu_percent": s["cpu_percent"],
                    "rss_mb": s["rss_mb"],
                    "num_threads": s["num_threads"],
                }
                for s in self._snapshots
            ],
        }
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    @staticmethod
    def _safe(func: Any) -> Any:
        try:
            value = func()
            if value is not None and value >= 0:
                return value
        except (psutil.Error, ValueError, AttributeError):
            pass
        return 0.0