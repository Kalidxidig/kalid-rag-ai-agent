# Copyright 2024
# Directory: yt-rag/app/services/rag.py

"""
RAG (Retrieval-Augmented Generation) service implementation.
Orchestrates the complete RAG pipeline: chunk → embed → search → generate.
"""

import logging
import time
import re
import os
from typing import List, Dict, Any

from ..core.database import db
from .embedding import embedding_service
from .chat import chat_service
from .chunker import chunker
from ..loaders.pdf_loader import load_pdf

logger = logging.getLogger(__name__)


class RAGService:
    """Main RAG service orchestrating the complete pipeline."""

    def __init__(self):
        self.db = db
        self.embedding_service = embedding_service
        self.chat_service = chat_service
        self.chunker = chunker


    async def seed_documents(
        self,
        documents: List[Dict[str, str]] = None
    ) -> int:

        start_time = time.time()


        if documents is None:

            documents = []

            documents_folder = "documents"


            for filename in os.listdir(documents_folder):

                if filename.lower().endswith(".pdf"):

                    path = os.path.join(
                        documents_folder,
                        filename
                    )

                    logger.info(
                        f"Loading PDF: {filename}"
                    )


                    pages = load_pdf(path)


                    for page in pages:

                        documents.append(
                            {
                                "chunk_id": f"{filename}_page_{page['page']}",
                                "source": filename,
                                "page": page["page"],
                                "text": page["text"]
                            }
                        )


        logger.info(
            f"Loaded {len(documents)} document sections"
        )


        try:

            chunks = self.chunker.chunk_documents(
                documents
            )


            logger.info(
                f"Created {len(chunks)} chunks"
            )


            texts = [
                chunk["text"]
                for chunk in chunks
            ]


            embeddings = await self.embedding_service.embed_texts(
                texts
            )


            for chunk, embedding in zip(
                chunks,
                embeddings
            ):

                chunk["embedding"] = embedding


            inserted_count = await self.db.upsert_chunks(
                chunks
            )


            elapsed_ms = int(
                (time.time() - start_time) * 1000
            )


            logger.info(
                f"Seeding completed in {elapsed_ms}ms"
            )


            return inserted_count


        except Exception as e:

            logger.error(
                f"Seeding failed: {e}"
            )

            raise



    async def answer_query(
        self,
        query: str,
        top_k: int = 6
    ) -> Dict[str, Any]:

        start_time = time.time()


        try:

            query_embedding = await self.embedding_service.embed_query(
                query
            )


            search_results = await self.db.vector_search(
                query_embedding,
                top_k
            )


            if not search_results:

                return {

                    "text":
                    "I don't have enough information to answer this question.",

                    "citations": [],

                    "debug":
                    {
                        "top_doc_ids": [],
                        "latency_ms":
                        int(
                            (time.time() - start_time) * 1000
                        )
                    }
                }



            context_blocks = self._prepare_context(
                search_results
            )


            answer_text = await self.chat_service.generate_answer(
                query,
                context_blocks
            )


            citations = self._extract_citations(
                answer_text,
                context_blocks
            )


            elapsed_ms = int(
                (time.time() - start_time) * 1000
            )


            return {

                "text": answer_text,

                "citations": citations,

                "sources": self._extract_sources(
                    context_blocks
                ),

                "debug":
                {
                    "top_doc_ids":
                    [
                        block["chunk_id"]
                        for block in context_blocks
                    ],

                    "latency_ms": elapsed_ms
                }
            }



        except Exception as e:

            logger.error(
                f"Query processing failed: {e}"
            )


            return {

                "text":
                "Sorry, I couldn't process your request right now. Please try again later.",

                "citations": [],

                "sources": [],

                "debug":
                {
                    "top_doc_ids": [],
                    "latency_ms":
                    int(
                        (time.time() - start_time) * 1000
                    )
                }
            }



    def _prepare_context(
        self,
        search_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:


        seen_prefixes = set()

        context_blocks = []


        for result in search_results:


            chunk_id = result.get(
                "chunk_id",
                ""
            )


            base_id = (
                chunk_id.split("#")[0]
                if "#" in chunk_id
                else chunk_id
            )


            if base_id not in seen_prefixes:

                context_blocks.append(
                    result
                )

                seen_prefixes.add(
                    base_id
                )


            if len(context_blocks) >= 4:
                break


        return context_blocks



    def _extract_citations(
    self,
    answer_text: str,
    context_blocks: List[Dict[str, Any]]
) -> List[str]:
    """
    Extract source citations from generated answer.
    Supports [chunk_id] format and automatically adds used sources.
    """

    citations = []

    # 1. Check explicit citations from AI response
    pattern = r"\[([^\]]+)\]"

    found = re.findall(
        pattern,
        answer_text
    )

    valid_ids = {
        block["chunk_id"]
        for block in context_blocks
    }

    for cite in found:
        if cite in valid_ids and cite not in citations:
            citations.append(cite)


    # 2. If AI does not include citations,
    # use retrieved context sources automatically
    if not citations:
        for block in context_blocks:
            source = block.get(
                "source",
                block.get("chunk_id", "unknown")
            )

            if source not in citations:
                citations.append(source)


    return citations



    def _extract_sources(
        self,
        context_blocks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:


        sources = []


        for block in context_blocks:

            sources.append(
                {
                    "file":
                    block.get(
                        "source",
                        "unknown"
                    ),

                    "page":
                    block.get(
                        "page",
                        None
                    ),

                    "chunk_id":
                    block.get(
                        "chunk_id",
                        "unknown"
                    )
                }
            )


        return sources



rag_service = RAGService()