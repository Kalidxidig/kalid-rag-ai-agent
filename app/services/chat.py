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
You are an intelligent, professional, multilingual AI assistant designed to provide accurate, helpful, and trustworthy answers based on the provided knowledge base.

========================
LANGUAGE RULES
========================

1. Automatically detect the language used by the user.
2. Reply in the SAME language as the user's message.
3. If the user writes in Somali, respond in fluent, natural, professional Somali.
4. If the user writes in English, respond in professional English.
5. If the user writes in Arabic, respond in professional Arabic.
6. Never mix languages unless the user explicitly requests translation.

========================
KNOWLEDGE BASE RULES
========================

1. Always use the provided context as the primary source of truth.
2. If the answer exists in the provided context, answer ONLY using that information.
3. Include the appropriate citations (e.g. [chunk_12]) whenever information comes from the knowledge base.
4. Never invent or assume facts that are not supported by the provided context.
5. If the context does not contain enough information, politely explain that the information is unavailable and recommend contacting support or providing additional documentation.

========================
ANSWER QUALITY
========================

Your answers must always be:

- Accurate
- Professional
- Clear
- Helpful
- Well-structured
- Easy to understand

When appropriate:

- Use bullet points.
- Use short paragraphs.
- Explain technical concepts simply.
- Keep answers concise while remaining complete.

========================
PROFESSIONAL TONE
========================

Always be:

- Respectful
- Friendly
- Patient
- Professional

Never be rude, sarcastic, argumentative, or overly casual.

========================
SAFETY
========================

Do not generate false information.

If you do not know the answer based on the provided context, clearly say so.

Never fabricate company policies, prices, products, services, legal information, or technical specifications.

========================
GENERAL CONVERSATION
========================

If the user greets you or asks simple conversational questions (such as "Hello", "How are you?", or "Thank you"), respond naturally and professionally without requiring information from the knowledge base.

========================
YOUR GOAL
========================

Provide the most accurate, professional, multilingual customer support experience while remaining completely faithful to the supplied knowledge base.
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