
import numpy as np
import pytest

from backend.app.vector_store.faiss_store import FaissVectorStore
from backend.app.vector_store.metadata_store import SQLiteMetadataStore
from backend.app.vector_store.vector_service import VectorRetrievalService


class FakeEmbeddingService:
    """Provide deterministic embeddings for unit tests."""

    model = None

    def __init__(self, query_vector=None):
        self.query_vector = (
            np.array([1, 0, 0], dtype=np.float32)
            if query_vector is None
            else np.asarray(query_vector, dtype=np.float32)
        )

    def embed_query(self, query):
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Invalid query.")

        return self.query_vector


@pytest.fixture
def retrieval_setup(tmp_path):
    vector_store = FaissVectorStore(dimension=3)

    metadata_store = SQLiteMetadataStore(
        tmp_path / "metadata.db"
    )

    embeddings = np.array([
        [1, 0, 0],
        [0.8, 0.2, 0],
        [0, 1, 0],
        [0, 0, 1]
    ], dtype=np.float32)

    vector_ids = [1001, 1002, 1003, 1004]

    vector_store.add_vectors(embeddings, vector_ids)

    chunks = [
        {
            "source_id": "doc_001",
            "chunk_index": 0,
            "page_number": 1,
            "language": "english",
            "text": "Identity card renewal requirements."
        },
        {
            "source_id": "doc_001",
            "chunk_index": 1,
            "page_number": 2,
            "language": "english",
            "text": "Required documents for renewal."
        },
        {
            "source_id": "doc_002",
            "chunk_index": 0,
            "page_number": 1,
            "language": "urdu",
            "text": "شناختی کارڈ کی تجدید کے لیے ضروری دستاویزات"
        },
        {
            "source_id": "doc_003",
            "chunk_index": 0,
            "page_number": 1,
            "language": "sindhi",
            "text": "سڃاڻپ ڪارڊ جي تجديد لاءِ ضروري دستاويز"
        }
    ]

    for vector_id, chunk in zip(vector_ids, chunks):
        metadata_store.add_chunk(vector_id, chunk)

    embedding_service = FakeEmbeddingService()

    service = VectorRetrievalService(
        embedding_service,
        vector_store,
        metadata_store
    )

    return service


# Activity M3.4 - Basic Retrieval

def test_retrieve_results(retrieval_setup):
    results = retrieval_setup.search("identity card renewal")

    assert len(results) == 4
    assert results[0]["vector_id"] == 1001
    assert results[0]["score"] == pytest.approx(1.0)


def test_results_are_ranked(retrieval_setup):
    results = retrieval_setup.search("identity card renewal")

    scores = [result["score"] for result in results]

    assert scores == sorted(scores, reverse=True)


def test_top_k(retrieval_setup):
    results = retrieval_setup.search(
        "identity card renewal",
        top_k=2
    )

    assert len(results) == 2


def test_top_k_larger_than_index(retrieval_setup):
    results = retrieval_setup.search(
        "identity card renewal",
        top_k=10
    )

    assert len(results) == 4


def test_result_contains_source_metadata(retrieval_setup):
    result = retrieval_setup.search("identity card renewal")[0]

    assert result["document_id"] == "doc_001"
    assert result["page_number"] == 1
    assert result["language"] == "english"
    assert result["text"] == "Identity card renewal requirements."
    assert "score" in result


# Activity M3.4 - Metadata Filtering

def test_filter_by_document(retrieval_setup):
    results = retrieval_setup.search(
        "identity card renewal",
        document_id="doc_001"
    )

    assert len(results) == 2
    assert all(r["document_id"] == "doc_001" for r in results)


def test_filter_by_language(retrieval_setup):
    results = retrieval_setup.search(
        "identity card renewal",
        language="urdu"
    )

    assert len(results) == 1
    assert results[0]["language"] == "urdu"


def test_filter_by_page(retrieval_setup):
    results = retrieval_setup.search(
        "identity card renewal",
        page_number=2
    )

    assert len(results) == 1
    assert results[0]["page_number"] == 2


def test_combined_filters(retrieval_setup):
    results = retrieval_setup.search(
        "identity card renewal",
        document_id="doc_001",
        language="english",
        page_number=2
    )

    assert len(results) == 1
    assert results[0]["vector_id"] == 1002


def test_filter_does_not_lose_relevant_candidates(retrieval_setup):
    results = retrieval_setup.search(
        "identity card renewal",
        top_k=1,
        language="sindhi"
    )

    assert len(results) == 1
    assert results[0]["language"] == "sindhi"


def test_no_matching_filter_results(retrieval_setup):
    results = retrieval_setup.search(
        "identity card renewal",
        document_id="unknown_document"
    )

    assert results == []


# Activity M3.4 - Validation

def test_empty_query(retrieval_setup):
    with pytest.raises(ValueError):
        retrieval_setup.search("  ")


def test_non_string_query(retrieval_setup):
    with pytest.raises(TypeError):
        retrieval_setup.search(123)


def test_invalid_top_k(retrieval_setup):
    with pytest.raises(ValueError):
        retrieval_setup.search("test", top_k=0)


def test_invalid_top_k_type(retrieval_setup):
    with pytest.raises(TypeError):
        retrieval_setup.search("test", top_k=True)


def test_invalid_language_filter(retrieval_setup):
    with pytest.raises(ValueError):
        retrieval_setup.search("test", language="french")


def test_invalid_page_filter(retrieval_setup):
    with pytest.raises(ValueError):
        retrieval_setup.search("test", page_number=0)


def test_empty_document_filter(retrieval_setup):
    with pytest.raises(ValueError):
        retrieval_setup.search("test", document_id=" ")


def test_empty_index_returns_empty_results(tmp_path):
    vector_store = FaissVectorStore(dimension=3)

    metadata_store = SQLiteMetadataStore(
        tmp_path / "empty.db"
    )

    service = VectorRetrievalService(
        FakeEmbeddingService(),
        vector_store,
        metadata_store
    )

    assert service.search("test") == []
