# Copyright 2024
# Directory: yt-rag/app/loaders/pdf_loader.py

"""
PDF loader for extracting text with page metadata.
Used for better RAG citations.
"""

from pypdf import PdfReader
from typing import List, Dict, Any


def load_pdf(file_path: str) -> List[Dict[str, Any]]:
    """
    Extract text from PDF file with page information.

    Returns:
        List of dictionaries:
        [
            {
                "text": "page content",
                "page": 1
            }
        ]
    """

    reader = PdfReader(file_path)

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):

        page_text = page.extract_text()

        if page_text:

            pages.append(
                {
                    "text": page_text,
                    "page": page_number
                }
            )

    return pages