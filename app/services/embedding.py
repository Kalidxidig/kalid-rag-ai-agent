import logging
from typing import List
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Local Sentence Transformer embedding service."""

    def __init__(self):
        self.model = None

    def load_model(self):
        """Load model only when needed."""
        if self.model is None:
            logger.info("Loading embedding model...")
            self.model = SentenceTransformer(
                "all-MiniLM-L6-v2"
            )
            logger.info("Embedding model loaded")


    async def embed_texts(
        self, texts: List[str]
    ) -> List[List[float]]:

        try:
            self.load_model()

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