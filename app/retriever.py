"""
retriever.py — Hybrid retrieval: FAISS dense + BM25 sparse, fused via Reciprocal Rank Fusion.

RRF formula: score(d) = Σ 1 / (k + rank(d))   where k=60 (empirically optimal).
Top-20 merged results are passed to the reranker.
"""

import logging
import numpy as np
from typing import List

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Combines dense (FAISS cosine) and sparse (BM25) retrieval via RRF."""

    def __init__(self, indexer, top_k: int = 30):
        self.indexer = indexer
        self.top_k = top_k
        self.faiss = indexer.faiss
        self.bm25 = indexer.bm25

    def retrieve(self, query: str) -> List[dict]:
        """
        Hybrid search: dense + sparse → RRF → top-20 chunks for reranking.
        Falls back to dense-only if BM25 is unavailable.
        """
        # 1. Dense Search (FAISS cosine similarity on normalized vectors)
        query_vec = self.indexer.model.encode(
            [query], normalize_embeddings=True
        ).astype(np.float32)
        dense_distances, dense_indices = self.faiss.search(query_vec, self.top_k)

        rrf_scores: dict[int, float] = {}
        k = 60

        for rank, idx in enumerate(dense_indices[0]):
            if idx == -1:
                continue
            rrf_scores[int(idx)] = rrf_scores.get(int(idx), 0.0) + 1.0 / (k + rank + 1)

        # 2. Sparse Search (BM25)
        if self.bm25 is not None:
            tokenized_query = query.lower().split()
            sparse_scores = self.bm25.get_scores(tokenized_query)
            sparse_indices = np.argsort(sparse_scores)[::-1][: self.top_k]

            for rank, idx in enumerate(sparse_indices):
                rrf_scores[int(idx)] = rrf_scores.get(int(idx), 0.0) + 1.0 / (k + rank + 1)

        # 3. Sort by RRF score and return top-20 for reranking
        sorted_indices = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)
        results = []
        for idx in sorted_indices[:20]:
            chunk_data = self.indexer.get_chunk_by_id(idx)
            if chunk_data:
                results.append(chunk_data)

        return results

    def retrieve_dense_only(self, query: str) -> List[dict]:
        """Dense-only retrieval (Experiment 1 baseline)."""
        query_vec = self.indexer.model.encode(
            [query], normalize_embeddings=True
        ).astype(np.float32)
        _, dense_indices = self.faiss.search(query_vec, self.top_k)
        results = []
        for idx in dense_indices[0]:
            if idx == -1:
                continue
            chunk_data = self.indexer.get_chunk_by_id(int(idx))
            if chunk_data:
                results.append(chunk_data)
        return results[:20]

    def refresh(self):
        """Refreshes references after live index update."""
        self.faiss = self.indexer.faiss
        self.bm25 = self.indexer.bm25
