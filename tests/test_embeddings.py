import numpy as np
import pytest
from src.embeddings.clip_encoder import CLIPEmbeddingPipeline


def test_encode_text_dimensions():
    pipeline = CLIPEmbeddingPipeline()
    text_vec = pipeline.encode_text("operador sem capacete de protecao")

    assert isinstance(text_vec, list)
    assert len(text_vec) == 512
    # Ensure vector is normalized to unit length
    norm = np.linalg.norm(np.array(text_vec, dtype=np.float32))
    assert pytest.approx(norm, 0.01) == 1.0


def test_encode_image_dimensions():
    pipeline = CLIPEmbeddingPipeline()
    dummy_image = np.zeros((224, 224, 3), dtype=np.uint8)
    img_vec = pipeline.encode_image(dummy_image)

    assert isinstance(img_vec, list)
    assert len(img_vec) == 512
    norm = np.linalg.norm(np.array(img_vec, dtype=np.float32))
    assert pytest.approx(norm, 0.01) == 1.0


def test_cosine_similarity():
    vec_a = [1.0, 0.0, 0.0]
    vec_b = [1.0, 0.0, 0.0]
    vec_c = [0.0, 1.0, 0.0]

    # Identical vectors -> similarity = 1.0
    sim_identical = CLIPEmbeddingPipeline.cosine_similarity(vec_a, vec_b)
    assert pytest.approx(sim_identical, 0.001) == 1.0

    # Orthogonal vectors -> similarity = 0.0
    sim_orthogonal = CLIPEmbeddingPipeline.cosine_similarity(vec_a, vec_c)
    assert pytest.approx(sim_orthogonal, 0.001) == 0.0
