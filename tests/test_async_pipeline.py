import asyncio
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.db import init_database
from src.embedding.provider import OllamaEmbeddingProvider
from src.generation.provider import OllamaLLMProvider
from src.pipeline.async_orchestrator import AsyncPipelineOrchestrator
from src.retrieval.hybrid import HybridRetriever


@pytest.fixture
def mock_embedder():
    embedder = MagicMock(spec=OllamaEmbeddingProvider)
    embedder.embed_batch = AsyncMock(return_value=[[0.1] * 768, [0.2] * 768])
    embedder.embed = AsyncMock(return_value=[0.1] * 768)
    embedder.close = AsyncMock()
    return embedder


@pytest.fixture
def mock_llm():
    llm = MagicMock(spec=OllamaLLMProvider)
    llm.generate_batch = AsyncMock(return_value=["4", "9"])
    llm.generate = AsyncMock(return_value="4")
    llm.close = AsyncMock()
    return llm


@pytest.fixture
def mock_retriever():
    retriever = MagicMock(spec=HybridRetriever)
    retriever.retrieve = AsyncMock(return_value=[])
    return retriever


@pytest.fixture
def temp_db(tmp_path):
    db_path = tmp_path / "test.db"
    init_database(str(db_path))
    return str(db_path)


@pytest.mark.asyncio
async def test_async_embedder_batch(mock_embedder):
    texts = ["2 + 2 = ?", "3 * 3 = ?"]
    embeddings = await mock_embedder.embed_batch(texts)
    assert len(embeddings) == 2
    assert all(len(e) == 768 for e in embeddings)
    mock_embedder.embed_batch.assert_called_once_with(texts)


@pytest.mark.asyncio
async def test_async_llm_batch(mock_llm):
    prompts = ["prompt1", "prompt2"]
    answers = await mock_llm.generate_batch(prompts)
    assert len(answers) == 2
    assert all(isinstance(a, str) for a in answers)
    mock_llm.generate_batch.assert_called_once_with(prompts)


@pytest.mark.asyncio
async def test_async_orchestrator_ingest_phase(
    temp_db, tmp_path, mock_embedder, mock_llm, mock_retriever
):
    # Create test CSV
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("""ID,Question,Comparação
1,2 + 2 = ?,4
2,3 * 3 = ?,9
""", encoding="utf-8")

    with patch("src.pipeline.async_orchestrator.OllamaEmbeddingProvider", return_value=mock_embedder), \
         patch("src.pipeline.async_orchestrator.OllamaLLMProvider", return_value=mock_llm), \
         patch("src.pipeline.async_orchestrator.HybridRetriever", return_value=mock_retriever), \
         patch("src.pipeline.async_orchestrator.check_ollama_sync"), \
         patch("src.pipeline.async_orchestrator.settings") as mock_settings:

        mock_settings.db_path = temp_db
        mock_settings.embed_batch_size = 32
        mock_settings.checkpoint_path = Path(temp_db).with_suffix(".checkpoint")
        mock_settings.checkpoint_interval = 50

        orchestrator = AsyncPipelineOrchestrator()
        orchestrator.embedder = mock_embedder
        orchestrator.llm = mock_llm
        orchestrator.retriever = mock_retriever

        await orchestrator._phase_ingest_async(csv_file)

        # Verify embeddings were called
        assert mock_embedder.embed_batch.called


@pytest.mark.asyncio
async def test_async_orchestrator_answer_phase(
    temp_db, tmp_path, mock_embedder, mock_llm, mock_retriever
):
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("""ID,Question,Comparação
1,2 + 2 = ?,
2,3 * 3 = ?,
""", encoding="utf-8")

    # Mock unanswered questions
    from src.models.question import Question
    from src.models.enums import DifficultyLevel

    unanswered = [
        Question(id=1, text="2 + 2 = ?", level=DifficultyLevel.SIMPLE),
        Question(id=2, text="3 * 3 = ?", level=DifficultyLevel.SIMPLE),
    ]

    with patch("src.pipeline.async_orchestrator.OllamaEmbeddingProvider", return_value=mock_embedder), \
         patch("src.pipeline.async_orchestrator.OllamaLLMProvider", return_value=mock_llm), \
         patch("src.pipeline.async_orchestrator.HybridRetriever", return_value=mock_retriever), \
         patch("src.pipeline.async_orchestrator.check_ollama_sync"), \
         patch("src.pipeline.async_orchestrator.settings") as mock_settings:

        mock_settings.db_path = temp_db
        mock_settings.gen_concurrency = 2
        mock_settings.top_k = 3
        mock_settings.checkpoint_path = Path(temp_db).with_suffix(".checkpoint")
        mock_settings.checkpoint_interval = 50

        orchestrator = AsyncPipelineOrchestrator()
        orchestrator.embedder = mock_embedder
        orchestrator.llm = mock_llm
        orchestrator.retriever = mock_retriever
        orchestrator.qrepo.get_unanswered = MagicMock(return_value=unanswered)

        await orchestrator._phase_answer_async()

        # Verify generation was called
        assert mock_llm.generate_batch.called


@pytest.mark.asyncio
async def test_async_orchestrator_answer_phase_no_rag(
    temp_db, tmp_path, mock_embedder, mock_llm, mock_retriever
):
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("""ID,Question,Comparação
1,2 + 2 = ?,
2,3 * 3 = ?,
""", encoding="utf-8")

    from src.models.question import Question
    from src.models.enums import DifficultyLevel

    unanswered = [
        Question(id=1, text="2 + 2 = ?", level=DifficultyLevel.SIMPLE),
        Question(id=2, text="3 * 3 = ?", level=DifficultyLevel.SIMPLE),
    ]

    with patch("src.pipeline.async_orchestrator.OllamaEmbeddingProvider", return_value=mock_embedder), \
         patch("src.pipeline.async_orchestrator.OllamaLLMProvider", return_value=mock_llm), \
         patch("src.pipeline.async_orchestrator.HybridRetriever", return_value=mock_retriever), \
         patch("src.pipeline.async_orchestrator.check_ollama_sync"), \
         patch("src.pipeline.async_orchestrator.settings") as mock_settings:

        mock_settings.db_path = temp_db
        mock_settings.gen_concurrency = 2
        mock_settings.top_k = 3
        mock_settings.use_rag = False
        mock_settings.checkpoint_path = Path(temp_db).with_suffix(".checkpoint")
        mock_settings.checkpoint_interval = 50

        orchestrator = AsyncPipelineOrchestrator()
        orchestrator.embedder = mock_embedder
        orchestrator.llm = mock_llm
        orchestrator.retriever = mock_retriever
        orchestrator.qrepo.get_unanswered = MagicMock(return_value=unanswered)

        await orchestrator._phase_answer_async()

        # Retrieval must NOT be called when --no-rag
        mock_retriever.retrieve.assert_not_called()
        assert mock_llm.generate_batch.called


def test_no_rag_ingest_stores_rows_without_calling_embedding_service(
    temp_db, tmp_path, mock_embedder, mock_llm, mock_retriever
):
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("ID,Question,Comparação\n1,2 + 2 = ?,4\n", encoding="utf-8")

    with patch(
        "src.pipeline.async_orchestrator.OllamaEmbeddingProvider", return_value=mock_embedder
    ), patch("src.pipeline.async_orchestrator.OllamaLLMProvider", return_value=mock_llm), patch(
        "src.pipeline.async_orchestrator.HybridRetriever", return_value=mock_retriever
    ), patch("src.pipeline.async_orchestrator.settings") as mock_settings:
        mock_settings.db_path = temp_db
        mock_settings.checkpoint_path = Path(temp_db).with_suffix(".checkpoint")
        mock_settings.checkpoint_interval = 50
        orchestrator = AsyncPipelineOrchestrator()
        orchestrator._phase_ingest_questions(csv_file)

    mock_embedder.embed.assert_not_called()
    mock_embedder.embed_batch.assert_not_called()
    assert len(orchestrator.qrepo.get_all()) == 1
