import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from queue import Queue

import sqlite_vec

from src.config import settings


class ConnectionManager:
    def __init__(self, db_path: str | None = None, pool_size: int = 4):
        self.db_path = db_path or settings.db_path
        self.pool_size = pool_size
        self._pool: Queue[sqlite3.Connection] = Queue(maxsize=pool_size)
        self._init_pool()

    def _init_pool(self) -> None:
        for _ in range(self.pool_size):
            conn = self._create_connection()
            self._pool.put(conn)

    def _create_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=-32768")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.row_factory = sqlite3.Row
        return conn

    @contextmanager
    def connection(self) -> Generator[sqlite3.Connection, None, None]:
        conn = self._pool.get()
        try:
            yield conn
        finally:
            self._pool.put(conn)

    def close_all(self) -> None:
        while not self._pool.empty():
            conn = self._pool.get()
            conn.close()
