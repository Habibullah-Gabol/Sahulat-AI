
from typing import Sequence

import numpy as np


class MultilingualEmbeddingService:
    """
    Generate dense multilingual embeddings using BGE-M3.

    The model is loaded explicitly and reused across requests.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str | None = None,
        max_seq_length: int = 512,
        model=None
    ):
        if max_seq_length < 1:
            raise ValueError("max_seq_length must be positive.")

        self.model_name = model_name
        self.device = device
        self.max_seq_length = max_seq_length
        self.model = model

        if self.model is not None:
            self.model.max_seq_length = max_seq_length

    def load(self):
        """Load the embedding model if it is not already loaded."""

        if self.model is not None:
            return

        import torch
        from sentence_transformers import SentenceTransformer

        if self.device is None:
            self.device = (
                "cuda" if torch.cuda.is_available() else "cpu"
            )

        self.model = SentenceTransformer(
            self.model_name,
            device=self.device
        )

        self.model.max_seq_length = self.max_seq_length

    @property
    def embedding_dimension(self) -> int:
        """Return the model's embedding dimension."""

        if self.model is None:
            raise RuntimeError("Load the model before requesting its dimension.")

        return self.model.get_embedding_dimension()

    def embed_texts(
        self,
        texts: Sequence[str],
        batch_size: int = 32,
        normalize_embeddings: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for a batch of texts.

        Returns:
            NumPy array with shape (number_of_texts, embedding_dimension).
        """

        if isinstance(texts, str) or not isinstance(texts, (list, tuple)):
            raise TypeError("texts must be a list or tuple of strings.")

        if not texts:
            raise ValueError("texts cannot be empty.")

        if any(not isinstance(text, str) for text in texts):
            raise TypeError("Every item must be a string.")

        if any(not text.strip() for text in texts):
            raise ValueError("Texts cannot contain empty strings.")

        if not isinstance(batch_size, int) or isinstance(batch_size, bool):
            raise TypeError("batch_size must be an integer.")

        if batch_size < 1:
            raise ValueError("batch_size must be positive.")

        if not isinstance(normalize_embeddings, bool):
            raise TypeError("normalize_embeddings must be a boolean.")

        if self.model is None:
            self.load()

        embeddings = self.model.encode(
            list(texts),
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=False
        )

        embeddings = np.asarray(embeddings, dtype=np.float32)

        expected_shape = (
            len(texts),
            self.embedding_dimension
        )

        if embeddings.shape != expected_shape:
            raise RuntimeError(
                f"Unexpected embedding shape: {embeddings.shape}. "
                f"Expected: {expected_shape}."
            )

        if not np.isfinite(embeddings).all():
            raise RuntimeError("Embeddings contain invalid numerical values.")

        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """Generate one embedding vector for a search query."""

        if not isinstance(query, str):
            raise TypeError("query must be a string.")

        if not query.strip():
            raise ValueError("query cannot be empty.")

        return self.embed_texts([query])[0]
