
import faiss
import numpy as np
import pytest

from backend.app.vector_store.faiss_store import FaissVectorStore


# Activity M3.1 - Index Initialization

def test_default_dimension():
    store = FaissVectorStore()
    assert store.dimension == 1024
    assert store.index.d == 1024


def test_custom_dimension():
    store = FaissVectorStore(384)
    assert store.dimension == 384


def test_index_type():
    store = FaissVectorStore()
    assert isinstance(store.index, faiss.IndexIDMap2)


def test_initial_vector_count():
    store = FaissVectorStore()
    assert store.total_vectors == 0


def test_invalid_dimension_type():
    with pytest.raises(TypeError):
        FaissVectorStore("1024")


def test_boolean_dimension_rejected():
    with pytest.raises(TypeError):
        FaissVectorStore(True)


def test_zero_dimension():
    with pytest.raises(ValueError):
        FaissVectorStore(0)


def test_negative_dimension():
    with pytest.raises(ValueError):
        FaissVectorStore(-10)


def test_reset_empty_index():
    store = FaissVectorStore()
    store.reset()
    assert store.total_vectors == 0


# Activity M3.2 - Vector Storage

def test_add_single_vector():
    store = FaissVectorStore(3)
    vectors = np.array([[1, 2, 3]], dtype=np.float32)

    store.add_vectors(vectors, [1001])

    assert store.total_vectors == 1
    assert store.has_id(1001)


def test_add_multiple_vectors():
    store = FaissVectorStore(3)
    vectors = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ], dtype=np.float32)

    store.add_vectors(vectors, [1001, 1002, 1003])

    assert store.total_vectors == 3


def test_vectors_are_normalized():
    store = FaissVectorStore(3)
    vectors = np.array([[3, 4, 0]], dtype=np.float32)

    store.add_vectors(vectors, [1001])

    stored = store.get_vector(1001)
    assert np.isclose(np.linalg.norm(stored), 1.0)


def test_get_vector_by_id():
    store = FaissVectorStore(3)
    vectors = np.array([[1, 2, 3]], dtype=np.float32)

    store.add_vectors(vectors, [1001])

    stored = store.get_vector(1001)
    expected = vectors[0] / np.linalg.norm(vectors[0])

    np.testing.assert_allclose(stored, expected)


def test_missing_vector_id():
    store = FaissVectorStore(3)

    with pytest.raises(KeyError):
        store.get_vector(9999)


def test_duplicate_existing_id():
    store = FaissVectorStore(3)
    vectors = np.array([[1, 0, 0]], dtype=np.float32)

    store.add_vectors(vectors, [1001])

    with pytest.raises(ValueError):
        store.add_vectors(vectors, [1001])


def test_duplicate_ids_in_batch():
    store = FaissVectorStore(3)
    vectors = np.array([
        [1, 0, 0],
        [0, 1, 0]
    ], dtype=np.float32)

    with pytest.raises(ValueError):
        store.add_vectors(vectors, [1001, 1001])


def test_invalid_vector_dimension():
    store = FaissVectorStore(3)
    vectors = np.array([[1, 2, 3, 4]], dtype=np.float32)

    with pytest.raises(ValueError):
        store.add_vectors(vectors, [1001])


def test_vector_id_count_mismatch():
    store = FaissVectorStore(3)
    vectors = np.array([[1, 0, 0]], dtype=np.float32)

    with pytest.raises(ValueError):
        store.add_vectors(vectors, [1001, 1002])


def test_zero_vector_rejected():
    store = FaissVectorStore(3)
    vectors = np.array([[0, 0, 0]], dtype=np.float32)

    with pytest.raises(ValueError):
        store.add_vectors(vectors, [1001])


def test_non_finite_vector_rejected():
    store = FaissVectorStore(3)
    vectors = np.array([[np.nan, 0, 1]], dtype=np.float32)

    with pytest.raises(ValueError):
        store.add_vectors(vectors, [1001])


def test_non_numeric_vector_rejected():
    store = FaissVectorStore(3)
    vectors = np.array([["a", "b", "c"]])

    with pytest.raises(TypeError):
        store.add_vectors(vectors, [1001])


def test_invalid_id_type():
    store = FaissVectorStore(3)
    vectors = np.array([[1, 0, 0]], dtype=np.float32)

    with pytest.raises(TypeError):
        store.add_vectors(vectors, ["1001"])


def test_reset_clears_vectors_and_ids():
    store = FaissVectorStore(3)
    vectors = np.array([[1, 0, 0]], dtype=np.float32)

    store.add_vectors(vectors, [1001])
    store.reset()

    assert store.total_vectors == 0
    assert not store.has_id(1001)


def test_failed_batch_does_not_add_vectors():
    store = FaissVectorStore(3)
    vectors = np.array([
        [1, 0, 0],
        [0, 1, 0]
    ], dtype=np.float32)

    with pytest.raises(ValueError):
        store.add_vectors(vectors, [1001, 1001])

    assert store.total_vectors == 0

# Activity M3.5 - FAISS Persistence

def test_save_index(tmp_path):
    store = FaissVectorStore(3)

    vectors = np.array([
        [1, 0, 0],
        [0, 1, 0]
    ], dtype=np.float32)

    store.add_vectors(vectors, [1001, 1002])

    file_path = tmp_path / "vector_store" / "faiss.index"

    store.save(str(file_path))

    assert file_path.exists()


def test_load_index(tmp_path):
    store = FaissVectorStore(3)

    vectors = np.array([
        [1, 0, 0],
        [0, 1, 0]
    ], dtype=np.float32)

    store.add_vectors(vectors, [1001, 1002])

    file_path = tmp_path / "faiss.index"
    store.save(str(file_path))

    new_store = FaissVectorStore(3)
    new_store.load(str(file_path))

    assert new_store.total_vectors == 2
    assert new_store.has_id(1001)
    assert new_store.has_id(1002)


def test_loaded_index_retrieves_vector(tmp_path):
    store = FaissVectorStore(3)

    vectors = np.array([[3, 4, 0]], dtype=np.float32)

    store.add_vectors(vectors, [1001])

    file_path = tmp_path / "faiss.index"
    store.save(str(file_path))

    new_store = FaissVectorStore(3)
    new_store.load(str(file_path))

    expected = np.array([0.6, 0.8, 0], dtype=np.float32)

    np.testing.assert_allclose(
        new_store.get_vector(1001),
        expected,
        atol=1e-6
    )


def test_loaded_index_supports_search(tmp_path):
    store = FaissVectorStore(3)

    vectors = np.array([
        [1, 0, 0],
        [0, 1, 0]
    ], dtype=np.float32)

    store.add_vectors(vectors, [1001, 1002])

    file_path = tmp_path / "faiss.index"
    store.save(str(file_path))

    new_store = FaissVectorStore(3)
    new_store.load(str(file_path))

    results = new_store.search(
        np.array([1, 0, 0], dtype=np.float32),
        top_k=1
    )

    assert results[0]["vector_id"] == 1001


def test_missing_index_file(tmp_path):
    store = FaissVectorStore(3)

    with pytest.raises(FileNotFoundError):
        store.load(str(tmp_path / "missing.index"))


def test_dimension_mismatch_on_load(tmp_path):
    store = FaissVectorStore(3)

    file_path = tmp_path / "faiss.index"
    store.save(str(file_path))

    different_store = FaissVectorStore(4)

    with pytest.raises(ValueError):
        different_store.load(str(file_path))


def test_invalid_save_path():
    store = FaissVectorStore(3)

    with pytest.raises(ValueError):
        store.save(" ")


def test_invalid_load_path():
    store = FaissVectorStore(3)

    with pytest.raises(ValueError):
        store.load(" ")
