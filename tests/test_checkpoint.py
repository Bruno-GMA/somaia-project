import pytest
from pathlib import Path
from src.pipeline.checkpoint import CheckpointManager


def test_checkpoint_manager(tmp_path):
    cp_path = tmp_path / "checkpoint.txt"
    cp = CheckpointManager(cp_path, interval=2)

    assert cp.is_processed(1) is False
    cp.mark_processed(1)
    assert cp.is_processed(1) is True

    cp.mark_processed(2)
    assert cp.get_processed_count() == 2

    # New instance should load from file
    cp2 = CheckpointManager(cp_path, interval=2)
    assert cp2.is_processed(1) is True
    assert cp2.is_processed(2) is True
    assert cp2.is_processed(3) is False


def test_checkpoint_flush_interval(tmp_path):
    cp_path = tmp_path / "checkpoint.txt"
    cp = CheckpointManager(cp_path, interval=3)

    cp.mark_processed(1)
    cp.mark_processed(2)
    # Not flushed yet (interval=3)

    cp2 = CheckpointManager(cp_path, interval=3)
    # File wasn't flushed, so might not be loaded
    # Force flush
    cp.force_flush()

    cp3 = CheckpointManager(cp_path, interval=3)
    assert cp3.is_processed(1) is True
    assert cp3.is_processed(2) is True