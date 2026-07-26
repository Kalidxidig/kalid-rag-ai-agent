"""
Keyword search service.
Finds documents using exact text matching.
"""

import logging
from typing import List, Dict, Any

from ..core.database import db


logger = logging.getLogger(__name__)


class KeywordSearch:

    async def search(
        self,
        query: str,
        limit: int = 6
    ) -> List[Dict[str, Any]]:


        try:

            client = db.get_client(
                admin=True
            )


            result = (
                client
                .table("rag_chunks")
                .select("*")
                .ilike(
                    "text",
                    f"%{query}%"
                )
                .limit(limit)
                .execute()
            )


            return result.data or []


        except Exception as e:

            logger.error(
                f"Keyword search failed: {e}"
            )

            return []



keyword_search = KeywordSearch()