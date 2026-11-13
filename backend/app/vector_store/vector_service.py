
from backend.app.vector_store.faiss_store import FaissVectorStore
from backend.app.vector_store.metadata_store import SQLiteMetadataStore
from backend.app.nlp.embedding_service import MultilingualEmbeddingService


class VectorRetrievalService:
    """Retrieve document chunks using multilingual semantic search."""

    def __init__(
        self,
        embedding_service: MultilingualEmbeddingService,
        vector_store: FaissVectorStore,
        metadata_store: SQLiteMetadataStore
    ):
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.metadata_store = metadata_store

        if (
            self.embedding_service.model is not None
            and self.embedding_service.embedding_dimension
            != self.vector_store.dimension
        ):
            raise ValueError("Embedding and FAISS dimensions do not match.")

    def search(
        self,
        query: str,
        top_k: int = 5,
        document_id: str | None = None,
        language: str | None = None,
        page_number: int | None = None
    ) -> list[dict]:
        """Retrieve ranked document chunks with optional filters."""

        if not isinstance(query, str):
            raise TypeError("Query must be a string.")

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if not isinstance(top_k, int) or isinstance(top_k, bool):
            raise TypeError("top_k must be an integer.")

        if top_k < 1:
            raise ValueError("top_k must be positive.")

        # Validate filters before performing model inference.
        if document_id is not None:
            if not isinstance(document_id, str) or not document_id.strip():
                raise ValueError("document_id must be a non-empty string.")

        if language is not None:
            if language not in {"english", "urdu", "sindhi"}:
                raise ValueError("Unsupported language.")

        if page_number is not None:
            if (
                not isinstance(page_number, int)
                or isinstance(page_number, bool)
                or page_number < 1
            ):
                raise ValueError("page_number must be a positive integer.")

        if self.vector_store.total_vectors == 0:
            return []

        query_embedding = self.embedding_service.embed_query(query)

        # Search the complete index to avoid losing filtered matches.
        candidates = self.vector_store.search(
            query_embedding,
            top_k=self.vector_store.total_vectors
        )

        results = []

        for candidate in candidates:
            metadata = self.metadata_store.get_chunk(
                candidate["vector_id"]
            )

            # Ignore vectors without corresponding metadata.
            if metadata is None:
                continue

            if (
                document_id is not None
                and metadata["document_id"] != document_id.strip()
            ):
                continue

            if (
                language is not None
                and metadata["language"] != language
            ):
                continue

            if (
                page_number is not None
                and metadata["page_number"] != page_number
            ):
                continue

            results.append({
                **metadata,
                "score": candidate["score"]
            })

            if len(results) == top_k:
                break

        return results
