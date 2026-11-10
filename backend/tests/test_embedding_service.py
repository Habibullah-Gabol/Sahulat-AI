
import numpy as np
import pytest

from backend.app.nlp.embedding_service import (
    MultilingualEmbeddingService
)


class FakeEmbeddingModel:

    def __init__(self, dimension=1024):
        self.dimension = dimension
        self.max_seq_length = None

    def get_embedding_dimension(self):
        return self.dimension

    def encode(
        self,
        texts,
        batch_size=32,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False
    ):
        vectors = np.ones(
            (len(texts), self.dimension),
            dtype=np.float32
        )

        if normalize_embeddings:
            vectors /= np.linalg.norm(
                vectors,
                axis=1,
                keepdims=True
            )

        return vectors


@pytest.fixture
def service():
    fake_model = FakeEmbeddingModel()

    return MultilingualEmbeddingService(
        model=fake_model
    )


def test_model_initialization(service):
    assert service.model is not None
    assert service.embedding_dimension == 1024


def test_max_sequence_length(service):
    assert service.model.max_seq_length == 512


def test_embedding_shape(service):
    result = service.embed_texts([
        "Hello Sahulat AI",
        "How are you?"
    ])

    assert result.shape == (2, 1024)


def test_single_query_embedding(service):
    result = service.embed_query("Hello world")

    assert result.shape == (1024,)


def test_embedding_normalization(service):
    result = service.embed_texts(["Hello world"])

    norm = np.linalg.norm(result[0])

    assert np.isclose(norm, 1.0)


def test_empty_text_list(service):
    with pytest.raises(ValueError):
        service.embed_texts([])


def test_invalid_text_input(service):
    with pytest.raises(TypeError):
        service.embed_texts("Hello")


def test_non_string_item(service):
    with pytest.raises(TypeError):
        service.embed_texts(["Hello", 123])


def test_empty_string_rejected(service):
    with pytest.raises(ValueError):
        service.embed_texts(["Hello", "   "])


def test_invalid_batch_size(service):
    with pytest.raises(ValueError):
        service.embed_texts(["Hello"], batch_size=0)


def test_invalid_batch_size_type(service):
    with pytest.raises(TypeError):
        service.embed_texts(["Hello"], batch_size=1.5)


def test_invalid_normalization_option(service):
    with pytest.raises(TypeError):
        service.embed_texts(
            ["Hello"],
            normalize_embeddings="yes"
        )


def test_invalid_query(service):
    with pytest.raises(ValueError):
        service.embed_query("")


def test_model_must_be_loaded_for_dimension():
    service = MultilingualEmbeddingService()

    with pytest.raises(RuntimeError):
        _ = service.embedding_dimension
