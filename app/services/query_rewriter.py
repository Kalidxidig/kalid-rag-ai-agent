# Copyright 2024
# Directory: yt-rag/app/services/query_rewriter.py

"""
Query rewriting service for conversational RAG.
Transforms follow-up questions into standalone questions.
"""

import logging
from typing import List, Dict

from .chat import chat_service

logger = logging.getLogger(__name__)


class QueryRewriter:
    """
    Converts conversational questions into standalone queries.
    """

    async def rewrite(
        self,
        query: str,
        history: List[Dict[str, str]]
    ) -> str:

        if not history:
            return query


        conversation = ""

        for message in history[-6:]:

            role = message.get(
                "role",
                "user"
            )

            content = message.get(
                "content",
                ""
            )

            conversation += (
                f"{role}: {content}\n"
            )


        prompt = f"""
Conversation history:

{conversation}


Current question:

{query}


Rewrite the current question into a standalone question.

Rules:
- Resolve pronouns like "they", "them", "their", "it", "this", "that" using the conversation history.
- Include important names, documents, topics, and entities from previous messages.
- Do not remove important context.
- If the question is already clear, return it unchanged.

Return ONLY the rewritten question.

Standalone question:
"""


        try:

            rewritten = await chat_service.simple_completion(
                prompt
            )

            rewritten = rewritten.strip()

            logger.info(
                f"Original query: {query}"
            )

            logger.info(
                f"Rewritten query: {rewritten}"
            )

            return rewritten


        except Exception as e:

            logger.error(
                f"Query rewriting failed: {e}"
            )

            return query



query_rewriter = QueryRewriter()