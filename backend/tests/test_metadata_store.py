
import sqlite3
import pytest

from backend.app.vector_store.metadata_store import SQLiteMetadataStore


@pytest.fixture
def store(tmp_path):
    db_path = tmp_path / "metadata.db"
    return SQLiteMetadataStore(db_path)


def sample_chunk(
    source_id="doc_001",
    chunk_index=0,
    page_number=1,
    language="english",
    text="This is a sample document passage."
):
    return {
        "source_id": source_id,
        "chunk_index": chunk_index,
        "page_number": page_number,
        "language": language,
        "text": text
    }


# Activity M3.3 - Database Initialization

def test_database_created(store):
    assert store.db_path.exists()


def test_initial_chunk_count(store):
    assert store.total_chunks == 0


# Activity M3.3 - Metadata Insertion

def test_add_chunk(store):
    store.add_chunk(1001, sample_chunk())

    assert store.total_chunks == 1


def test_inserted_chunk_fields(store):
    store.add_chunk(1001, sample_chunk())

    result = store.get_chunk(1001)

    assert result["vector_id"] == 1001
    assert result["document_id"] == "doc_001"
    assert result["chunk_index"] == 0
    assert result["page_number"] == 1
    assert result["language"] == "english"
    assert result["text"] == "This is a sample document passage."
    assert result["char_count"] == len(result["text"])


def test_duplicate_vector_id_rejected(store):
    store.add_chunk(1001, sample_chunk())

    with pytest.raises(sqlite3.IntegrityError):
        store.add_chunk(1001, sample_chunk(chunk_index=1))


def test_missing_chunk_returns_none(store):
    assert store.get_chunk(9999) is None


# Activity M3.3 - Metadata Filtering

def test_filter_by_document(store):
    store.add_chunk(1001, sample_chunk("doc_001"))
    store.add_chunk(1002, sample_chunk("doc_002"))

    results = store.get_chunks(document_id="doc_001")

    assert len(results) == 1
    assert results[0]["vector_id"] == 1001


def test_filter_by_language(store):
    store.add_chunk(1001, sample_chunk(language="english"))
    store.add_chunk(
        1002,
        sample_chunk(
            source_id="doc_002",
            language="urdu",
            text="شناختی کارڈ کی تجدید"
        )
    )

    results = store.get_chunks(language="urdu")

    assert len(results) == 1
    assert results[0]["vector_id"] == 1002


def test_filter_by_page(store):
    store.add_chunk(1001, sample_chunk(page_number=1))
    store.add_chunk(1002, sample_chunk(page_number=2, chunk_index=1))

    results = store.get_chunks(page_number=2)

    assert len(results) == 1
    assert results[0]["vector_id"] == 1002


def test_combined_filters(store):
    store.add_chunk(1001, sample_chunk())
    store.add_chunk(
        1002,
        sample_chunk(
            source_id="doc_002",
            language="urdu"
        )
    )
    store.add_chunk(
        1003,
        sample_chunk(
            source_id="doc_001",
            language="urdu",
            chunk_index=1
        )
    )

    results = store.get_chunks(
        document_id="doc_001",
        language="urdu"
    )

    assert len(results) == 1
    assert results[0]["vector_id"] == 1003


def test_filter_without_parameters_returns_all(store):
    store.add_chunk(1001, sample_chunk())
    store.add_chunk(1002, sample_chunk(chunk_index=1))

    assert len(store.get_chunks()) == 2


# Activity M3.3 - Metadata Validation

def test_invalid_vector_id_type(store):
    with pytest.raises(TypeError):
        store.add_chunk("1001", sample_chunk())


def test_negative_vector_id(store):
    with pytest.raises(ValueError):
        store.add_chunk(-1, sample_chunk())


def test_invalid_chunk_type(store):
    with pytest.raises(TypeError):
        store.add_chunk(1001, "invalid")


def test_missing_document_id(store):
    chunk = sample_chunk()
    chunk["source_id"] = ""

    with pytest.raises(ValueError):
        store.add_chunk(1001, chunk)


def test_invalid_chunk_index(store):
    chunk = sample_chunk()
    chunk["chunk_index"] = -1

    with pytest.raises(ValueError):
        store.add_chunk(1001, chunk)


def test_invalid_page_number(store):
    chunk = sample_chunk()
    chunk["page_number"] = 0

    with pytest.raises(ValueError):
        store.add_chunk(1001, chunk)


def test_unsupported_language(store):
    chunk = sample_chunk()
    chunk["language"] = "french"

    with pytest.raises(ValueError):
        store.add_chunk(1001, chunk)


def test_empty_text(store):
    chunk = sample_chunk()
    chunk["text"] = "   "

    with pytest.raises(ValueError):
        store.add_chunk(1001, chunk)


def test_delete_chunk(store):
    store.add_chunk(1001, sample_chunk())

    assert store.delete_chunk(1001) is True
    assert store.total_chunks == 0


def test_delete_missing_chunk(store):
    assert store.delete_chunk(9999) is False
