from src.config import settings


def chunk_question(
    text: str, max_words: int | None = None, overlap: int | None = None
) -> list[str]:
    max_words = max_words or settings.chunk_threshold
    overlap = overlap or settings.chunk_overlap

    words = text.split()
    if len(words) <= max_words:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += max_words - overlap
    return chunks
