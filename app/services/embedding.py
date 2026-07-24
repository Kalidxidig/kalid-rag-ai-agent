import logging
from typing import List
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Local Sentence Transformer embedding service."""

    def __init__(self):
        self.model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2",
            device="cpu"
        )

    async def embed_texts(
        self,
        texts: List[str]
    ) -> List[List[float]]:

        try:
            embeddings = self.model.encode(
                texts,
                batch_size=8,
                convert_to_numpy=True,
                show_progress_bar=False
            )

            return embeddings.tolist()

        except Exception as e:
            logger.error(
                f"Failed to generate embeddings: {e}"
            )
            raise


    async def embed_query(
        self,
        query: str
    ) -> List[float]:

        embeddings = await self.embed_texts([query])
        return embeddings[0]


embedding_service = EmbeddingService()