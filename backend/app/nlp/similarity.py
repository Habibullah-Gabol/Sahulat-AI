
import numpy as np


def _validate_vector(vector: np.ndarray, name: str) -> np.ndarray:
    if not isinstance(vector, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array.")

    if vector.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")

    if vector.size == 0:
        raise ValueError(f"{name} cannot be empty.")

    if not np.issubdtype(vector.dtype, np.number):
        raise TypeError(f"{name} must contain numerical values.")

    if not np.isfinite(vector).all():
        raise ValueError(f"{name} contains non-finite values.")

    vector = vector.astype(np.float64, copy=False)

    if np.linalg.norm(vector) == 0:
        raise ValueError(f"{name} cannot be a zero vector.")

    return vector


def _validate_matrix(matrix: np.ndarray, name: str) -> np.ndarray:
    if not isinstance(matrix, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array.")

    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional matrix.")

    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} cannot be empty.")

    if not np.issubdtype(matrix.dtype, np.number):
        raise TypeError(f"{name} must contain numerical values.")

    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contains non-finite values.")

    matrix = matrix.astype(np.float64, copy=False)

    norms = np.linalg.norm(matrix, axis=1)

    if np.any(norms == 0):
        raise ValueError(f"{name} cannot contain zero vectors.")

    return matrix


def cosine_similarity(
    vector_a: np.ndarray,
    vector_b: np.ndarray
) -> float:
    """Calculate cosine similarity between two embedding vectors."""

    vector_a = _validate_vector(vector_a, "vector_a")
    vector_b = _validate_vector(vector_b, "vector_b")

    if vector_a.shape != vector_b.shape:
        raise ValueError("Embedding vectors must have the same dimensions.")

    similarity = np.dot(vector_a, vector_b) / (
        np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    )

    return float(np.clip(similarity, -1.0, 1.0))


def pairwise_similarity(
    embeddings_a: np.ndarray,
    embeddings_b: np.ndarray
) -> np.ndarray:
    """
    Calculate cosine similarity between two embedding collections.

    Returns a matrix with shape (len(embeddings_a), len(embeddings_b)).
    """

    embeddings_a = _validate_matrix(embeddings_a, "embeddings_a")
    embeddings_b = _validate_matrix(embeddings_b, "embeddings_b")

    if embeddings_a.shape[1] != embeddings_b.shape[1]:
        raise ValueError("Embedding dimensions must match.")

    norms_a = np.linalg.norm(embeddings_a, axis=1, keepdims=True)
    norms_b = np.linalg.norm(embeddings_b, axis=1, keepdims=True)

    normalized_a = embeddings_a / norms_a
    normalized_b = embeddings_b / norms_b

    similarities = normalized_a @ normalized_b.T

    return np.clip(similarities, -1.0, 1.0)


def rank_by_similarity(
    query_embedding: np.ndarray,
    document_embeddings: np.ndarray,
    top_k: int | None = None
) -> list[dict]:
    """
    Rank document embeddings by cosine similarity to a query.

    Returns:
        A list of dictionaries containing document index and score.
        Results are sorted from highest to lowest similarity.
    """

    query_embedding = _validate_vector(
        query_embedding,
        "query_embedding"
    )

    document_embeddings = _validate_matrix(
        document_embeddings,
        "document_embeddings"
    )

    if query_embedding.shape[0] != document_embeddings.shape[1]:
        raise ValueError("Query and document dimensions must match.")

    if top_k is not None:
        if not isinstance(top_k, int) or isinstance(top_k, bool):
            raise TypeError("top_k must be an integer or None.")

        if top_k < 1:
            raise ValueError("top_k must be positive.")

    similarities = pairwise_similarity(
        query_embedding.reshape(1, -1),
        document_embeddings
    )[0]

    ranked_indices = np.argsort(-similarities, kind="stable")

    if top_k is not None:
        ranked_indices = ranked_indices[:top_k]

    return [
        {
            "index": int(index),
            "score": float(similarities[index])
        }
        for index in ranked_indices
    ]
