"""
Persistent conversation memory for Xidig AI Assistant.
Stores chat history in Supabase database.
"""

from typing import List, Dict

from ..core.database import db


class ConversationMemory:
    """
    Database-backed conversation storage.
    """


    async def get_history(
        self,
        conversation_id: str
    ) -> List[Dict[str, str]]:

        client = db.get_client(admin=True)

        result = client.table(
            "conversation_messages"
        ).select(
            "role, content"
        ).eq(
            "conversation_id",
            conversation_id
        ).order(
            "created_at"
        ).execute()


        return result.data or []



    async def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str
    ):

        client = db.get_client(admin=True)

        client.table(
            "conversation_messages"
        ).insert(
            {
                "conversation_id": conversation_id,
                "role": role,
                "content": content
            }
        ).execute()



    async def clear_history(
        self,
        conversation_id: str
    ):

        client = db.get_client(admin=True)

        client.table(
            "conversation_messages"
        ).delete().eq(
            "conversation_id",
            conversation_id
        ).execute()



conversation_memory = ConversationMemory()