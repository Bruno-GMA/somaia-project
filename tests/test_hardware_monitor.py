import json

import pytest

from src.utils.hardware_monitor import HardwareMonitor


@pytest.fixture
def mocked_process(mocker):
    process = mocker.MagicMock()
    process.cpu_percent.return_value = 42.5
    process.memory_info.return_value = mocker.MagicMock(rss=512.3 * 1024 * 1024)
    process.num_threads.return_value = 8
    mocker.patch(
        "src.utils.hardware_monitor.psutil.Process",
        return_value=process,
    )
    return process


def test_snapshot_fields(mocker, mocked_process):
    monitor = HardwareMonitor()
    monitor.start()
    snapshot = monitor.snapshot()

    assert set(snapshot) == {
        "timestamp",
        "cpu_percent",
        "rss_mb",
        "num_threads",
        "elapsed_s",
    }
    assert snapshot["cpu_percent"] == 42.5
    assert snapshot["num_threads"] == 8
    assert snapshot["elapsed_s"] >= 0.0
    assert snapshot["rss_mb"] == pytest.approx(512.3, rel=1e-2)


def test_snapshot_sanitizes_negative_values(mocked_process):
    mocked_process.cpu_percent.return_value = -1.0
    monitor = HardwareMonitor()
    monitor.start()
    snapshot = monitor.snapshot()
    assert snapshot["cpu_percent"] == 0.0


def test_summarize(mocker, mocked_process):
    monitor = HardwareMonitor()
    monitor.start()
    monitor._snapshots = [
        {"cpu_percent": 10.0, "rss_mb": 100.0, "num_threads": 4},
        {"cpu_percent": 20.0, "rss_mb": 200.0, "num_threads": 6},
        {"cpu_percent": 30.0, "rss_mb": 150.0, "num_threads": 5},
    ]
    summary = monitor.summarize()

    assert summary["cpu_percent_mean"] == pytest.approx(20.0)
    assert summary["cpu_percent_peak"] == 30.0
    assert summary["rss_mb_peak"] == 200.0
    assert summary["num_threads_peak"] == 6
    assert summary["total_elapsed_s"] >= 0.0


def test_summarize_without_snapshots_calls_snapshot(mocked_process):
    monitor = HardwareMonitor()
    monitor.start()
    summary = monitor.summarize()

    assert summary["cpu_percent_peak"] == 42.5
    assert summary["rss_mb_peak"] == pytest.approx(512.3, rel=1e-2)


def test_format_report_contains_metrics(mocked_process):
    monitor = HardwareMonitor()
    monitor.start()
    report = monitor.format_report()

    assert "CPU média" in report
    assert "CPU pico" in report
    assert "RAM pico (RSS)" in report
    assert "Threads máximas" in report


def test_save_json_writes_parseable_file(tmp_path, mocker, mocked_process):
    monitor = HardwareMonitor()
    monitor.start()
    monitor._snapshots = [
        {"timestamp": 1694457600.0, "cpu_percent": 42.5, "rss_mb": 512.3, "num_threads": 8}
    ]

    path = tmp_path / "hardware_log.json"
    monitor.save_json(str(path))

    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert "summary" in data
    assert "snapshots" in data
    assert len(data["snapshots"]) == 1
    assert data["snapshots"][0]["cpu_percent"] == 42.5
    assert "elapsed_s" not in data["snapshots"][0]