import logging
from typing import List
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Local embedding service using Sentence Transformers."""

    def __init__(self):
        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

    async def embed_texts(
        self, texts: List[str]
    ) -> List[List[float]]:

        try:
            embeddings = self.model.encode(
                texts,
                normalize_embeddings=True
            )

            return embeddings.tolist()

        except Exception as e:
            logger.error(
                f"Failed to generate embeddings: {e}"
            )
            raise

    async def embed_query(
        self, query: str
    ) -> List[float]:

        embeddings = await self.embed_texts([query])
        return embeddings[0]


embedding_service = EmbeddingService()