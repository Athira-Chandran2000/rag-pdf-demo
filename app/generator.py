"""
generator.py — Groq LLM generation (llama-3.1-8b-instant via LPU hardware).

Groq free tier delivers 200–500ms end-to-end due to custom LPU inference hardware.
The system prompt enforces citation discipline: every claim must include [filename, Page N].
"""

import os
import logging
from groq import Groq

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a highly precise research assistant specializing in machine learning and AI.\n"
    "Use ONLY the provided context passages from ArXiv papers to answer the user's question.\n\n"
    "Rules:\n"
    "1. Answer ONLY using the provided context. If the answer is not present, say exactly: "
    "'I cannot find the answer in the provided documents.'\n"
    "2. For every factual claim, cite the source in brackets: [filename, Page N].\n"
    "3. Be concise and professional. Do not pad your answer.\n"
    "4. Do not refer to 'context' or 'passages' — call them 'the research papers' or 'the documents'.\n"
    "5. If multiple sources support a claim, cite all of them."
)


class GroqGenerator:
    """Wraps the Groq client to generate grounded, cited answers."""

    def __init__(self):
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            logger.warning("GROQ_API_KEY not found in environment — generation will fail.")
        self.client = Groq(api_key=api_key)

    def generate(self, question: str, chunks: list[dict]) -> str:
        """
        Generate a grounded answer from `chunks` for `question`.
        Returns answer string. On error, returns a descriptive error message.
        """
        if not chunks:
            return (
                "I'm sorry, I couldn't find any relevant information in the indexed documents "
                "to answer your question."
            )

        context_parts = []
        for i, c in enumerate(chunks, 1):
            source = c.get("source", "Unknown")
            page = c.get("page_number", "?")
            text = c.get("text", "")
            context_parts.append(f"[{i}] SOURCE: {source} (Page {page})\n{text}")

        context = "\n\n".join(context_parts)

        try:
            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
                ],
                max_tokens=800,
                temperature=0.1,  # Low temperature = high precision / reproducibility
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Groq generation error: {e}")
            return f"Error generating answer: {str(e)}"
