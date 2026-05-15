"""
pipeline.py — Orchestrates the full RAG pipeline: retrieval → reranking → parent swap → generation.

Latency is tracked at every stage and returned with the answer for transparency.
Supports three experiment modes:
  - dense_only=True,  skip_rerank=True  → Experiment 1 (Baseline)
  - dense_only=True,  skip_rerank=False → Experiment 2 (Dense + Reranker)
  - dense_only=False, skip_rerank=False → Experiment 3 (Hybrid + Reranker, Full)
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from app.indexer import Indexer
from app.retriever import HybridRetriever
from app.reranker import FlashRankReranker
from app.generator import GroqGenerator

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Production RAG pipeline with four sequential stages:
    retrieval → reranking → parent chunk swap → generation.
    """

    def __init__(
        self,
        retrieval_top_k: int = 20,
        rerank_top_n: int = 5,
        dense_only: bool = False,
        skip_rerank: bool = False,
    ):
        mode = (
            "dense-only, no-rerank" if dense_only and skip_rerank
            else "dense-only, rerank" if dense_only
            else "hybrid+rerank"
        )
        logger.info(f"Initialising RAGPipeline [{mode}]")

        self.indexer = Indexer()
        self.indexer.load_all()
        self.retriever = HybridRetriever(self.indexer, top_k=retrieval_top_k)
        self.reranker = FlashRankReranker(top_n=rerank_top_n)
        self.generator = GroqGenerator()

        self.dense_only = dense_only
        self.skip_rerank = skip_rerank
        self.retrieval_top_k = retrieval_top_k
        self.rerank_top_n = rerank_top_n

    def run(self, question: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        """
        Full pipeline run with per-stage latency tracking.

        Returns:
            {
                answer:          str,
                source_nodes:    list[dict],   # for UI display
                chunks_used:     list[dict],   # for RAGAS evaluation
                latencies:       dict[str, float],  # ms per stage
                total_latency_ms: float,
            }
        """
        effective_top_k = top_k or self.rerank_top_n
        latencies: Dict[str, float] = {}

        # ── Stage 1: Retrieval ────────────────────────────────────────────────
        t0 = time.perf_counter()
        if self.dense_only:
            initial_nodes = self.retriever.retrieve_dense_only(question)
        else:
            initial_nodes = self.retriever.retrieve(question)
        latencies["retrieval_ms"] = (time.perf_counter() - t0) * 1000

        # ── Stage 2: Reranking ────────────────────────────────────────────────
        t0 = time.perf_counter()
        if self.skip_rerank:
            reranked_nodes = initial_nodes[:effective_top_k]
        else:
            reranked_nodes = self.reranker.rerank(question, initial_nodes)
        latencies["reranking_ms"] = (time.perf_counter() - t0) * 1000

        # ── Stage 3: Parent Chunk Swap ────────────────────────────────────────
        t0 = time.perf_counter()
        final_nodes: List[dict] = []
        seen_parents: set = set()

        for node in reranked_nodes:
            parent_id = node.get("parent_id")
            if parent_id and parent_id not in seen_parents:
                parent_text = self.indexer.get_parent_text(parent_id)
                if parent_text:
                    # Replace child text with richer parent context
                    node = dict(node)  # don't mutate the original
                    node["text"] = parent_text
                    seen_parents.add(parent_id)
            final_nodes.append(node)
            if len(final_nodes) >= effective_top_k:
                break
        latencies["parent_swap_ms"] = (time.perf_counter() - t0) * 1000

        # ── Stage 4: Generation ───────────────────────────────────────────────
        t0 = time.perf_counter()
        answer = self.generator.generate(question, final_nodes)
        latencies["generation_ms"] = (time.perf_counter() - t0) * 1000

        total_ms = sum(latencies.values())
        latencies["total_ms"] = total_ms

        # Build citation list for the UI
        citations = [
            {
                "source": node.get("source", "Unknown"),
                "page": node.get("page_number", "?"),
            }
            for node in final_nodes
        ]

        return {
            "answer": answer,
            "citations": citations,
            "source_nodes": final_nodes,   # rich objects for UI
            "chunks_used": final_nodes,     # alias used by evaluator
            "latencies": latencies,
            "total_latency_ms": total_ms,
        }

    def query(self, question: str) -> Dict[str, Any]:
        """Alias for run() — used by experiment runner and evaluator."""
        return self.run(question)
