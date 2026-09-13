from pathlib import Path


class CheckpointManager:
    def __init__(self, path: Path, interval: int = 50):
        self.path = path
        self.interval = interval
        self._processed: set[int] = set()
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            with open(self.path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            self._processed.add(int(line))
                        except ValueError:
                            pass

    def is_processed(self, qid: int) -> bool:
        return qid in self._processed

    def mark_processed(self, qid: int) -> None:
        self._processed.add(qid)
        if len(self._processed) % self.interval == 0:
            self._flush()

    def _flush(self) -> None:
        with open(self.path, "w") as f:
            for qid in sorted(self._processed):
                f.write(f"{qid}\n")

    def force_flush(self) -> None:
        self._flush()

    def get_processed_count(self) -> int:
        return len(self._processed)
