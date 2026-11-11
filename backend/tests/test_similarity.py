
import numpy as np
import pytest

from backend.app.nlp.similarity import (
    cosine_similarity,
    pairwise_similarity,
    rank_by_similarity
)


def test_identical_vectors():
    a = np.array([1.0, 0.0, 0.0])
    assert cosine_similarity(a, a) == pytest.approx(1.0)


def test_orthogonal_vectors():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert cosine_similarity(a, b) == pytest.approx(0.0)


def test_opposite_vectors():
    a = np.array([1.0, 0.0])
    b = np.array([-1.0, 0.0])
    assert cosine_similarity(a, b) == pytest.approx(-1.0)


def test_unnormalized_vectors():
    a = np.array([2.0, 0.0])
    b = np.array([5.0, 0.0])
    assert cosine_similarity(a, b) == pytest.approx(1.0)


def test_mismatched_dimensions():
    with pytest.raises(ValueError):
        cosine_similarity(
            np.array([1.0, 2.0]),
            np.array([1.0, 2.0, 3.0])
        )


def test_zero_vector():
    with pytest.raises(ValueError):
        cosine_similarity(
            np.array([0.0, 0.0]),
            np.array([1.0, 0.0])
        )


def test_non_finite_vector():
    with pytest.raises(ValueError):
        cosine_similarity(
            np.array([1.0, np.nan]),
            np.array([1.0, 0.0])
        )


def test_wrong_input_type():
    with pytest.raises(TypeError):
        cosine_similarity([1.0, 0.0], np.array([1.0, 0.0]))


def test_pairwise_shape():
    a = np.array([[1.0, 0.0], [0.0, 1.0]])
    b = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]])

    result = pairwise_similarity(a, b)

    assert result.shape == (2, 3)


def test_pairwise_values():
    a = np.array([[1.0, 0.0], [0.0, 1.0]])
    b = np.array([[1.0, 0.0], [0.0, 1.0]])

    result = pairwise_similarity(a, b)

    expected = np.array([[1.0, 0.0], [0.0, 1.0]])
    np.testing.assert_allclose(result, expected, atol=1e-7)


def test_pairwise_rejects_mismatched_dimensions():
    with pytest.raises(ValueError):
        pairwise_similarity(
            np.array([[1.0, 0.0]]),
            np.array([[1.0, 0.0, 0.0]])
        )


def test_pairwise_rejects_zero_rows():
    with pytest.raises(ValueError):
        pairwise_similarity(
            np.array([[1.0, 0.0], [0.0, 0.0]]),
            np.array([[1.0, 0.0]])
        )


def test_document_ranking():
    query = np.array([1.0, 0.0])
    documents = np.array([
        [0.0, 1.0],
        [1.0, 0.0],
        [-1.0, 0.0]
    ])

    results = rank_by_similarity(query, documents)

    assert [item["index"] for item in results] == [1, 0, 2]
    assert results[0]["score"] == pytest.approx(1.0)


def test_top_k():
    query = np.array([1.0, 0.0])
    documents = np.array([
        [0.0, 1.0],
        [1.0, 0.0],
        [-1.0, 0.0]
    ])

    results = rank_by_similarity(query, documents, top_k=2)

    assert len(results) == 2
    assert results[0]["index"] == 1


def test_top_k_greater_than_document_count():
    query = np.array([1.0, 0.0])
    documents = np.array([[1.0, 0.0], [0.0, 1.0]])

    results = rank_by_similarity(query, documents, top_k=10)

    assert len(results) == 2


def test_invalid_top_k():
    query = np.array([1.0, 0.0])
    documents = np.array([[1.0, 0.0]])

    with pytest.raises(ValueError):
        rank_by_similarity(query, documents, top_k=0)

    with pytest.raises(TypeError):
        rank_by_similarity(query, documents, top_k=1.5)


def test_stable_tie_order():
    query = np.array([1.0, 0.0])
    documents = np.array([
        [0.0, 1.0],
        [0.0, 2.0],
        [1.0, 0.0]
    ])

    results = rank_by_similarity(query, documents)

    assert [item["index"] for item in results] == [2, 0, 1]
