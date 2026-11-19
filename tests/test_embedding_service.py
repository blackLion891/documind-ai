
import numpy as np
import pytest

from app.services.embedding_service import generate_embeddings


class FakeEmbeddingModel:
    def encode(self, texts, **kwargs):
        return np.array([
            [0.1] * 384 for _ in texts
        ])


def test_generate_embeddings(monkeypatch):
    from app.services import embedding_service

    monkeypatch.setattr(
        embedding_service,
        "get_embedding_model",
        lambda: FakeEmbeddingModel()
    )

    texts = [
        "Artificial intelligence is useful.",
        "Machine learning is a part of AI."
    ]

    vectors = generate_embeddings(texts)

    assert len(vectors) == 2
    assert len(vectors[0]) == 384
    assert len(vectors[1]) == 384


def test_empty_text_list():
    with pytest.raises(ValueError):
        generate_embeddings([])


def test_blank_text():
    with pytest.raises(ValueError):
        generate_embeddings(["   "])
