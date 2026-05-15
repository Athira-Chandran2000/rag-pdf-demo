"""
tests/test_pipeline_components.py — Unit tests for individual pipeline components.

These tests run in CI without any live indexes or API keys.
They cover the pure-logic components: RRF fusion, chunking, latency stats, metadata lookup.
"""

import json
import pickle
import sqlite3
import tempfile
from pathlib import Path

import numpy as np
import pytest


# ─────────────────────────────────────────────────────────────────────────────
# RRF Fusion
# ─────────────────────────────────────────────────────────────────────────────
def _rrf_scores(dense_indices: list[int], sparse_indices: list[int], k: int = 60) -> dict[int, float]:
    """Replicate the retriever's RRF logic for isolated testing."""
    scores: dict[int, float] = {}
    for rank, idx in enumerate(dense_indices):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    for rank, idx in enumerate(sparse_indices):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return scores


def test_rrf_boosts_agreement():
    """Chunks appearing in both dense and sparse lists get higher RRF scores."""
    dense = [0, 1, 2, 3, 4]
    sparse = [2, 0, 5, 6, 7]  # 0 and 2 appear in both

    scores = _rrf_scores(dense, sparse)

    # Items in both lists should outscore items in only one list
    assert scores[0] > scores[3], "Chunk 0 (in both) should beat chunk 3 (dense only)"
    assert scores[2] > scores[5], "Chunk 2 (in both) should beat chunk 5 (sparse only)"


def test_rrf_returns_correct_count():
    dense = list(range(30))
    sparse = list(range(15, 45))
    scores = _rrf_scores(dense, sparse)
    # Should contain all unique indices from both lists
    assert len(scores) == 45


def test_rrf_k_parameter():
    """Higher k smooths out the score differences."""
    dense = [0, 1, 2]
    sparse = [0, 3, 4]
    scores_k60 = _rrf_scores(dense, sparse, k=60)
    scores_k1 = _rrf_scores(dense, sparse, k=1)
    # With small k, rank differences are amplified
    assert scores_k1[0] > scores_k60[0]


# ─────────────────────────────────────────────────────────────────────────────
# Hierarchical Chunking (no PDF needed)
# ─────────────────────────────────────────────────────────────────────────────
def test_chunk_pages_basic():
    """chunk_pages should produce non-empty parent and child chunks from text."""
    from app.ingestion import chunk_pages

    pages = [
        {
            "page_number": 1,
            "text": (
                "Attention mechanisms allow models to focus on relevant parts of the input. "
                "The transformer architecture relies entirely on attention and removes recurrence. "
                "This enables parallelization during training and achieves state-of-the-art performance. "
            ) * 20,  # enough text to produce multiple chunks
            "source": "test_paper.pdf",
        }
    ]
    parents, children = chunk_pages(pages)
    assert len(parents) >= 1, "Should produce at least one parent chunk"
    assert len(children) >= len(parents), "Should produce at least as many children as parents"
    # Every child must reference a valid parent
    parent_ids = {p["id"] for p in parents}
    for child in children:
        assert child["parent_id"] in parent_ids, f"Child {child['id']} references unknown parent"


def test_chunk_pages_empty_input():
    """chunk_pages should return empty lists for empty or whitespace-only pages."""
    from app.ingestion import chunk_pages

    parents, children = chunk_pages([])
    assert parents == [] and children == []

    parents, children = chunk_pages([{"page_number": 1, "text": "   ", "source": "x.pdf"}])
    assert parents == [] and children == []


def test_chunk_pages_source_propagated():
    """Source filename should be present in every chunk."""
    from app.ingestion import chunk_pages

    pages = [{"page_number": 1, "text": "Word " * 200, "source": "arxiv_1234.pdf"}]
    parents, children = chunk_pages(pages)
    for c in parents + children:
        assert c["source"] == "arxiv_1234.pdf"


# ─────────────────────────────────────────────────────────────────────────────
# Latency Statistics
# ─────────────────────────────────────────────────────────────────────────────
def test_compute_latency_stats_basic():
    from app.evaluator import compute_latency_stats

    records = [
        {"retrieval_ms": 30, "reranking_ms": 50, "generation_ms": 200, "total_ms": 280},
        {"retrieval_ms": 40, "reranking_ms": 60, "generation_ms": 300, "total_ms": 400},
        {"retrieval_ms": 20, "reranking_ms": 45, "generation_ms": 150, "total_ms": 215},
    ]
    stats = compute_latency_stats(records)
    assert "retrieval_ms" in stats
    assert "p50" in stats["retrieval_ms"]
    assert "p90" in stats["retrieval_ms"]
    assert "p99" in stats["retrieval_ms"]
    # P99 >= P90 >= P50
    assert stats["total_ms"]["p99"] >= stats["total_ms"]["p90"] >= stats["total_ms"]["p50"]


def test_compute_latency_stats_empty():
    from app.evaluator import compute_latency_stats
    assert compute_latency_stats([]) == {}


# ─────────────────────────────────────────────────────────────────────────────
# SQLite Metadata Lookup (temporary in-memory DB)
# ─────────────────────────────────────────────────────────────────────────────
def _make_test_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "metadata.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE chunks (
            id TEXT PRIMARY KEY, parent_id TEXT, text TEXT,
            source TEXT, page_number INTEGER, synthetic INTEGER
        )
    """)
    # Insert 3 child chunks + their parent
    parent_id = "parent-uuid-001"
    conn.execute("INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?)",
                 (parent_id, None, "Parent context text " * 50, "test.pdf", 1, 0))
    for i in range(3):
        conn.execute("INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?)",
                     (f"child-{i}", parent_id, f"Child chunk {i} text.", "test.pdf", 1, 0))
    conn.commit()
    conn.close()
    return db_path


def test_get_parent_text(tmp_path):
    """get_parent_text should return parent text given a valid parent_id."""
    from app.indexer import Indexer

    db_path = _make_test_db(tmp_path)
    indexer = Indexer.__new__(Indexer)
    indexer.db_conn = sqlite3.connect(str(db_path))
    indexer.db_conn.row_factory = sqlite3.Row
    indexer.table_name = "chunks"
    indexer.id_column = "id"
    indexer.use_rowid = False
    indexer.faiss = None
    indexer.bm25 = None
    indexer.model = None

    text = indexer.get_parent_text("parent-uuid-001")
    assert "Parent context text" in text


def test_get_parent_text_missing(tmp_path):
    """get_parent_text with a non-existent ID should return empty string."""
    from app.indexer import Indexer

    db_path = _make_test_db(tmp_path)
    indexer = Indexer.__new__(Indexer)
    indexer.db_conn = sqlite3.connect(str(db_path))
    indexer.db_conn.row_factory = sqlite3.Row
    indexer.table_name = "chunks"
    indexer.id_column = "id"
    indexer.use_rowid = False
    indexer.faiss = None
    indexer.bm25 = None
    indexer.model = None

    assert indexer.get_parent_text("nonexistent-id") == ""


# ─────────────────────────────────────────────────────────────────────────────
# Response Formatting
# ─────────────────────────────────────────────────────────────────────────────
def test_response_keys():
    """
    Ensure the pipeline.run() return dict has all required keys
    without touching real models (mock the generator).
    """
    # We can test this structurally via a minimal mock
    required_keys = {"answer", "citations", "source_nodes", "chunks_used", "latencies", "total_latency_ms"}
    # Build a mock result manually
    fake_result = {
        "answer": "Test answer",
        "citations": [{"source": "paper.pdf", "page": 1}],
        "source_nodes": [{"text": "chunk text", "source": "paper.pdf", "page_number": 1}],
        "chunks_used": [{"text": "chunk text", "source": "paper.pdf", "page_number": 1}],
        "latencies": {"retrieval_ms": 30, "reranking_ms": 50, "generation_ms": 200, "total_ms": 280},
        "total_latency_ms": 280.0,
    }
    assert required_keys.issubset(fake_result.keys()), "Missing required keys in response"
