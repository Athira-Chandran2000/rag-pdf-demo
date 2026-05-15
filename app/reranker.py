"""
reranker.py — Flashrank cross-encoder reranking.

Flashrank uses a quantized ms-marco-MiniLM cross-encoder (~60MB, CPU-native).
Expected latency: 30–60ms on CPU for 20 candidates, <15ms on GPU.
"""

import logging
from flashrank import Ranker, RerankRequest

logger = logging.getLogger(__name__)


class FlashRankReranker:
    """Reranks candidate chunks using a lightweight cross-encoder model."""

    def __init__(self, top_n: int = 5):
        self.top_n = top_n
        logger.info("Loading Flashrank ranker (ms-marco-MiniLM-L-12-v2)...")
        self.ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")
        logger.info("Flashrank loaded.")

    def rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        """
        Rerank `candidates` against `query`. Returns top_n sorted by relevance.
        Each candidate must have a 'text' key.
        """
        if not candidates:
            return []

        passages = [{"id": i, "text": c.get("text", ""), "meta": c} for i, c in enumerate(candidates)]
        request = RerankRequest(query=query, passages=passages)
        results = self.ranker.rerank(request)

        # Return the original chunk dicts (including all metadata) sorted by score
        return [r["meta"] for r in results[: self.top_n]]
