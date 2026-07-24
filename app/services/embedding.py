from sentence_transformers import SentenceTransformer

class EmbeddingService:

    def __init__(self):
        self.model = SentenceTransformer(
            "sentence-transformers/paraphrase-MiniLM-L3-v2"
        )

    async def embed_texts(self, texts):
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            batch_size=8
        )
        return embeddings.tolist()

    async def embed_query(self, query):
        embeddings = await self.embed_texts([query])
        return embeddings[0]


embedding_service = EmbeddingService()