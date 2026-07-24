import logging
from typing import List
from huggingface_hub import InferenceClient

from ..core.config import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()


class EmbeddingService:

    def __init__(self):
        self.client = InferenceClient(
            token=settings.hf_token
        )

        self.model = "sentence-transformers/all-MiniLM-L6-v2"


    async def embed_texts(
        self,
        texts: List[str]
    ) -> List[List[float]]:

        try:
            embeddings = []

            for text in texts:
                result = self.client.feature_extraction(
                    text,
                    model=self.model
                )

                embeddings.append(result.tolist())

            return embeddings

        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise


    async def embed_query(
        self,
        query: str
    ) -> List[float]:

        embeddings = await self.embed_texts([query])
        return embeddings[0]


embedding_service = EmbeddingService()