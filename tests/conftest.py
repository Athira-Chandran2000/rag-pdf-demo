"""
tests/conftest.py — Shared pytest fixtures.
"""

import json
import sqlite3
import tempfile
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="session")
def sample_pages():
    """Ten pages of synthetic ML text, enough to produce multiple chunks."""
    return [
        {
            "page_number": i + 1,
            "text": (
                f"Page {i+1}. The transformer model uses multi-head self-attention to process "
                "sequences in parallel. Each attention head learns different relationships between "
                "tokens. The feed-forward network applies non-linear transformations. "
                "Layer normalization stabilizes training. "
            ) * 10,
            "source": "test_transformer_paper.pdf",
        }
        for i in range(10)
    ]


@pytest.fixture(scope="session")
def temp_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("rag_test")
