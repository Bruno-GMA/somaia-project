import json

import pytest

from src.utils.hardware_monitor import HardwareMonitor


@pytest.fixture
def mocked_process(mocker):
    process = mocker.MagicMock(pid=123)
    process.cpu_percent.return_value = 42.5
    process.memory_info.return_value = mocker.MagicMock(rss=512 * 1024 * 1024)
    process.num_threads.return_value = 8
    mocker.patch("src.utils.hardware_monitor.psutil.Process", return_value=process)
    mocker.patch("src.utils.hardware_monitor.psutil.cpu_percent", return_value=31.0)
    mocker.patch(
        "src.utils.hardware_monitor.psutil.virtual_memory",
        return_value=mocker.MagicMock(used=1024 * 1024 * 1024),
    )
    return process


def test_snapshot_has_explicit_scopes_and_no_energy_claim(mocked_process):
    monitor = HardwareMonitor()
    monitor.start(run_id="r1", mode="rag")
    snapshot = monitor.sample("generation")
    assert snapshot["schema_version"] == 2
    assert snapshot["stage"] == "generation"
    assert snapshot["process"]["scope"] == "python_process"
    assert snapshot["process"]["rss_mb"] == 512
    assert snapshot["system"]["scope"] == "system"
    assert snapshot["ollama_process"]["status"] == "not_collected"
    assert snapshot["energy_joules"] is None


def test_collection_failure_is_null_with_reason(mocked_process):
    mocked_process.memory_info.side_effect = OSError("no access")
    monitor = HardwareMonitor()
    monitor.start()
    snapshot = monitor.snapshot()
    assert snapshot["process"]["rss_mb"] is None
    assert snapshot["process"]["errors"]["rss_mb"] == "OSError"


def test_summary_and_json_preserve_observed_qualification(tmp_path, mocked_process):
    monitor = HardwareMonitor()
    monitor.start()
    monitor.sample("retrieval")
    summary = monitor.summarize()
    assert summary["n_samples"] == 1
    assert summary["rss_mb_peak_observed"] == 512
    path = tmp_path / "hardware.json"
    monitor.save_json(str(path))
    assert json.loads(path.read_text())["snapshots"][0]["stage"] == "retrieval"
