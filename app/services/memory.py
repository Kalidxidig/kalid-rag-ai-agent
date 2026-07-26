"""
Simple conversation memory storage for Xidig AI Assistant.
Stores temporary chat history per conversation.
"""

from typing import Dict, List


class ConversationMemory:
    """
    In-memory conversation storage.
    """

    def __init__(self):
        self.sessions: Dict[str, List[Dict[str, str]]] = {}


    def get_history(
        self,
        conversation_id: str
    ) -> List[Dict[str, str]]:

        return self.sessions.get(
            conversation_id,
            []
        )


    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str
    ):

        if conversation_id not in self.sessions:
            self.sessions[conversation_id] = []


        self.sessions[conversation_id].append(
            {
                "role": role,
                "content": content
            }
        )


    def clear_history(
        self,
        conversation_id: str
    ):

        if conversation_id in self.sessions:
            del self.sessions[conversation_id]



# Global memory instance
conversation_memory = ConversationMemory()