import logging
from typing import List
from openai import OpenAI
from ..core.config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """OpenAI embedding service."""

    def __init__(self):
        self.client = OpenAI(
            api_key=settings.openai_api_key
        )

    async def embed_texts(
        self, texts: List[str]
    ) -> List[List[float]]:

        try:
            response = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=texts
            )

            return [
                item.embedding
                for item in response.data
            ]

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