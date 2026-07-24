import logging
from typing import List
from openai import AsyncOpenAI

from ..core.config import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()


class EmbeddingService:
    """OpenAI embedding service."""

    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=settings.openai_api_key
        )
        self.model = settings.openai_embed_model


    async def embed_texts(
        self,
        texts: List[str]
    ) -> List[List[float]]:

        try:
            response = await self.client.embeddings.create(
                model=self.model,
                input=texts
            )

            embeddings = [
                item.embedding
                for item in response.data
            ]

            return embeddings

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