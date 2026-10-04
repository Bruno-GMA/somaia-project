from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="SOMAIA_",
        extra="ignore",
    )

    ollama_base_url: str = "http://localhost:11434"
    chat_model: str = "qwen3:1.7b"
    embed_model: str = "nomic-embed-text"
    embed_dim: int = 768

    db_path: str = "math_rag.db"

    use_rag: bool = True

    top_k: int = 3
    chunk_threshold: int = 256
    chunk_overlap: int = 32
    answer_column: str = "resposta"

    embed_batch_size: int = 32
    embed_concurrency: int = 4
    gen_concurrency: int = 2
    request_timeout: int = 120

    checkpoint_interval: int = 50
    # Optional per-run destination for monitor artifacts (used by paired evaluation).
    artifact_dir: str | None = None

    @property
    def db_path_obj(self) -> Path:
        return Path(self.db_path)

    @property
    def checkpoint_path(self) -> Path:
        return self.db_path_obj.with_suffix(".checkpoint")


settings = Settings()
