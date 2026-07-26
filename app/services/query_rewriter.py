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

        for message in history:

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
Keep the original meaning.
If the question is already clear, return it unchanged.

Standalone question:
"""


        try:

            rewritten = await chat_service.simple_completion(
                prompt
            )

            return rewritten.strip()


        except Exception as e:

            logger.error(
                f"Query rewriting failed: {e}"
            )

            return query



query_rewriter = QueryRewriter()