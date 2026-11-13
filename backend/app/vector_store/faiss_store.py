
import faiss
import numpy as np
from pathlib import Path

class FaissVectorStore:
    """Manage vectors using a FAISS inner-product index."""

    def __init__(self, dimension: int = 1024):
        if not isinstance(dimension, int) or isinstance(dimension, bool):
            raise TypeError("dimension must be an integer.")

        if dimension < 1:
            raise ValueError("dimension must be positive.")

        self.dimension = dimension
        base_index = faiss.IndexFlatIP(dimension)
        self.index = faiss.IndexIDMap2(base_index)
        self._ids = set()

    @property
    def total_vectors(self) -> int:
        return self.index.ntotal

    def reset(self) -> None:
        """Remove all stored vectors and IDs."""
        self.index.reset()
        self._ids.clear()

    def has_id(self, vector_id: int) -> bool:
        """Check whether a vector ID exists."""
        self._validate_id(vector_id)
        return vector_id in self._ids

    @staticmethod
    def _validate_id(vector_id: int) -> None:
        if not isinstance(vector_id, (int, np.integer)):
            raise TypeError("Vector ID must be an integer.")

        if isinstance(vector_id, (bool, np.bool_)):
            raise TypeError("Vector ID must be an integer.")

    def add_vectors(
        self,
        embeddings: np.ndarray,
        ids: list[int]
    ) -> None:
        """Add normalized embeddings with custom integer IDs."""

        if not isinstance(embeddings, np.ndarray):
            raise TypeError("Embeddings must be a NumPy array.")

        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D array.")

        if embeddings.shape[1] != self.dimension:
            raise ValueError("Embedding dimension mismatch.")

        if embeddings.shape[0] == 0:
            raise ValueError("Embeddings cannot be empty.")

        if not np.issubdtype(embeddings.dtype, np.number):
            raise TypeError("Embeddings must contain numeric values.")

        if not np.isrealobj(embeddings):
            raise TypeError("Embeddings must contain real values.")

        if not np.isfinite(embeddings).all():
            raise ValueError("Embeddings contain non-finite values.")

        if not isinstance(ids, (list, tuple)):
            raise TypeError("IDs must be a list or tuple.")

        if len(ids) != embeddings.shape[0]:
            raise ValueError("Number of IDs must match number of vectors.")

        for vector_id in ids:
            self._validate_id(vector_id)

        ids = [int(vector_id) for vector_id in ids]

        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate IDs found in input.")

        if any(vector_id in self._ids for vector_id in ids):
            raise ValueError("Vector ID already exists.")

        vectors = np.asarray(embeddings, dtype=np.float32)

        if not np.isfinite(vectors).all():
            raise ValueError("Embeddings exceed float32 numerical range.")

        norms = np.linalg.norm(vectors, axis=1, keepdims=True)

        if np.any(norms == 0):
            raise ValueError("Zero vectors cannot be stored.")

        vectors = vectors / norms

        vector_ids = np.asarray(ids, dtype=np.int64)

        self.index.add_with_ids(vectors, vector_ids)
        self._ids.update(ids)

    def get_vector(self, vector_id: int) -> np.ndarray:
        """Retrieve a stored vector by its custom ID."""

        self._validate_id(vector_id)

        if vector_id not in self._ids:
            raise KeyError(f"Vector ID {vector_id} does not exist.")

        return self.index.reconstruct(int(vector_id))

    
    def save(self, file_path: str) -> None:
    """Save the FAISS index to disk."""

      if not isinstance(file_path, str) or not file_path.strip():
          raise ValueError("file_path must be a non-empty string.")

      path = Path(file_path)
      path.parent.mkdir(parents=True, exist_ok=True)

      faiss.write_index(self.index, str(path))


    def load(self, file_path: str) -> None:
        """Load a saved FAISS index from disk."""

        if not isinstance(file_path, str) or not file_path.strip():
            raise ValueError("file_path must be a non-empty string.")

        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"FAISS index not found: {path}")

        loaded_index = faiss.read_index(str(path))

        if loaded_index.d != self.dimension:
            raise ValueError("Loaded index dimension mismatch.")

        if not isinstance(loaded_index, faiss.IndexIDMap2):
            raise ValueError("Saved index must use IndexIDMap2.")

        # Rebuild the ID set from the loaded index.
        ids = faiss.vector_to_array(loaded_index.id_map)

        self.index = loaded_index
        self._ids = set(int(vector_id) for vector_id in ids)

    def search(
    self,
    query_embedding: np.ndarray,
    top_k: int = 5
    ) -> list[dict]:
      """Search vectors using inner-product similarity."""

      if not isinstance(query_embedding, np.ndarray):
          raise TypeError("Query embedding must be a NumPy array.")

      if query_embedding.ndim != 1:
          raise ValueError("Query embedding must be one-dimensional.")

      if query_embedding.shape[0] != self.dimension:
          raise ValueError("Query embedding dimension mismatch.")

      if not np.issubdtype(query_embedding.dtype, np.number):
          raise TypeError("Query embedding must be numeric.")

      if not np.isrealobj(query_embedding):
          raise TypeError("Query embedding must contain real values.")

      if not np.isfinite(query_embedding).all():
          raise ValueError("Query embedding contains non-finite values.")

      if not isinstance(top_k, int) or isinstance(top_k, bool):
          raise TypeError("top_k must be an integer.")

      if top_k < 1:
          raise ValueError("top_k must be positive.")

      if self.total_vectors == 0:
          return []

      query = np.asarray(query_embedding, dtype=np.float32)

      if not np.isfinite(query).all():
          raise ValueError("Query embedding exceeds float32 numerical range.")

      norm = np.linalg.norm(query)

      if norm == 0:
          raise ValueError("Query embedding cannot be a zero vector.")

      query = (query / norm).reshape(1, -1)

      scores, ids = self.index.search(
          query,
          min(top_k, self.total_vectors)
      )

      results = []

      for score, vector_id in zip(scores[0], ids[0]):
          if vector_id == -1:
              continue

          results.append({
              "vector_id": int(vector_id),
              "score": float(score)
          })

      return results

