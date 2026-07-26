# Copyright 2024
# Directory: yt-rag/app/services/chat.py

"""
Chat completion service for generating RAG responses.
Supports OpenRouter, Anthropic and Gemini.
"""


import logging
from typing import List, Dict, Any

import openai
import anthropic
from google import genai

from ..core.config import get_settings


logger = logging.getLogger(__name__)

settings = get_settings()



class ChatService:
    """Service for chat completions."""


    def __init__(self):

        self.provider = settings.ai_provider


        # OpenRouter (OpenAI compatible)
        if self.provider == "openai":

            self.client = openai.OpenAI(
                api_key=settings.openrouter_api_key,
                base_url="https://openrouter.ai/api/v1"
            )

            self.model = settings.openai_chat_model



        elif self.provider == "anthropic":

            self.client = anthropic.Anthropic(
                api_key=settings.anthropic_api_key
            )

            self.model = settings.anthropic_chat_model



        elif self.provider == "gemini":

            self.client = genai.Client(
                api_key=settings.gemini_api_key
            )

            self.model = settings.gemini_chat_model



        else:
            raise ValueError(
                f"Unsupported AI provider: {self.provider}"
            )


        logger.info(
            f"Initialized chat service: {self.provider} - {self.model}"
        )




    async def generate_answer(
        self,
        query: str,
        context_blocks: List[Dict[str, Any]],
        conversation_history: List[Dict[str, str]] = None
    ) -> str:


        context_parts = []

        for block in context_blocks:

            chunk_id = block.get(
                "chunk_id",
                "unknown"
            )

            text = block.get(
                "text",
                ""
            )

            context_parts.append(
                f"[{chunk_id}] {text}"
            )


        context = "\n\n".join(context_parts)



        history_text = ""

        if conversation_history:

            history_parts = []

            for message in conversation_history:

                role = message.get(
                    "role",
                    "user"
                )

                content = message.get(
                    "content",
                    ""
                )

                history_parts.append(
                    f"{role}: {content}"
                )


            history_text = "\n\n".join(history_parts)



        system_prompt = """
You are Xidig AI Assistant.

Rules:
1. Answer using the provided context.
2. Use conversation history to understand follow-up questions.
3. Include citations like [chunk_id].
4. Be concise and professional.
5. If information is missing, say you do not know.
"""



        user_prompt = f"""
Previous Conversation:

{history_text}


Current Context:

{context}


Current Question:

{query}


Answer using the context and previous conversation.
"""


        try:


            if self.provider == "openai":


                logger.info(
                    f"SENDING MODEL TO API: {self.model}"
                )


                response = self.client.chat.completions.create(

                    model=self.model,

                    messages=[

                        {
                            "role": "system",
                            "content": system_prompt
                        },

                        {
                            "role": "user",
                            "content": user_prompt
                        }

                    ],

                    temperature=settings.temperature,

                    max_tokens=1000
                )


                answer = (
                    response
                    .choices[0]
                    .message
                    .content
                )



            elif self.provider == "anthropic":


                response = self.client.messages.create(

                    model=self.model,

                    max_tokens=1000,

                    temperature=settings.temperature,

                    system=system_prompt,

                    messages=[
                        {
                            "role": "user",
                            "content": user_prompt
                        }
                    ]

                )


                answer = response.content[0].text



            elif self.provider == "gemini":


                response = self.client.models.generate_content(

                    model=self.model,

                    contents=f"""
{system_prompt}

{user_prompt}
"""
                )


                answer = response.text



            logger.info(
                "Answer generated successfully"
            )


            return answer or "No answer generated."



        except Exception as e:

            logger.error(
                f"Failed to generate answer: {e}"
            )

            return (
                "I encountered an error while processing your question: "
                f"{str(e)}"
            )




chat_service = ChatService()