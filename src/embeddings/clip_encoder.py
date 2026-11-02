import hashlib
import logging
import os
from typing import List, Union
import numpy as np
from PIL import Image

logger = logging.getLogger("omnivision.embeddings")


class CLIPEmbeddingPipeline:
    """Multimodal encoder that maps text and images into a shared 512d latent space."""

    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "laion2b_s34b_b79k"):
        self.model_name = model_name
        self.pretrained = pretrained
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        self.device = "cpu"
        self.is_mock = False
        self._initialize_model()

    def _initialize_model(self):
        """Attempts to load OpenCLIP weights, with graceful fallback to deterministic mock."""
        try:
            import torch
            import open_clip

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                self.model_name,
                pretrained=self.pretrained,
                device=self.device
            )
            self.tokenizer = open_clip.get_tokenizer(self.model_name)
            self.model.eval()
            logger.info("OpenCLIP model %s (%s) loaded on device: %s", self.model_name, self.pretrained, self.device)
        except Exception as exc:
            logger.warning("OpenCLIP model could not be initialized (%s). Operating in mock mode.", exc)
            self.is_mock = True

    def encode_text(self, text: str) -> List[float]:
        """Encodes descriptive natural language query into normalized 512d vector."""
        if self.is_mock or self.model is None:
            return self._generate_deterministic_vector(text)

        import torch
        tokens = self.tokenizer([text]).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            return text_features.cpu().numpy()[0].tolist()

    def encode_image(self, image_input: Union[str, np.ndarray, Image.Image]) -> List[float]:
        """Encodes image frame into normalized 512d vector."""
        if self.is_mock or self.model is None:
            seed_key = str(image_input) if isinstance(image_input, str) else "image_array"
            return self._generate_deterministic_vector(seed_key)

        import torch
        if isinstance(image_input, str):
            pil_img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            pil_img = Image.fromarray(image_input)
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
        else:
            raise TypeError("Unsupported image input type.")

        image_tensor = self.preprocess(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image_tensor)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            return image_features.cpu().numpy()[0].tolist()

    @staticmethod
    def _generate_deterministic_vector(seed: str, dim: int = 512) -> List[float]:
        """Generates reproducible unit-length vector from hash for local/test execution."""
        hash_digest = hashlib.sha256(seed.encode("utf-8")).digest()
        # Seed numpy RNG with integer hash
        seed_int = int.from_bytes(hash_digest[:4], "big")
        rng = np.random.default_rng(seed_int)
        vec = rng.standard_normal(dim).astype(np.float32)
        vec /= np.linalg.norm(vec)
        return vec.tolist()

    @staticmethod
    def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
        """Calculates cosine similarity between two normalized vectors."""
        a = np.array(vec_a, dtype=np.float32)
        b = np.array(vec_b, dtype=np.float32)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
