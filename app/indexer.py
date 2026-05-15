"""
indexer.py — Loads FAISS, BM25, SQLite indexes at startup and builds them from chunks.json.

Key design decisions:
- Adaptive schema detection: handles any column naming convention produced by build scripts.
- Physical rowid fallback: if FAISS ntotal != DB row count, falls back to rowid (0-indexed offset).
- CHUNKS_FILE reference is defined here so run_indexing() and build_indexes.py share the same path.
"""

import os
import json
import pickle
import logging
import sqlite3
from pathlib import Path

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
CHUNKS_FILE = PROCESSED_DIR / "chunks.json"
FAISS_INDEX_FILE = PROCESSED_DIR / "faiss.index"
METADATA_DB_FILE = PROCESSED_DIR / "metadata.db"
BM25_FILE = PROCESSED_DIR / "bm25.pkl"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


class Indexer:
    """Loads and exposes FAISS, BM25, SQLite, and the embedding model."""

    def __init__(self):
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.faiss = None
        self.bm25 = None
        self.db_conn = None
        self.table_name = "chunks"
        self.id_column = "id"
        self.use_rowid = False

    def load_all(self):
        """Load all three indexes from disk. Safe to call even if files are absent."""
        if FAISS_INDEX_FILE.exists():
            self.faiss = faiss.read_index(str(FAISS_INDEX_FILE))
            logger.info(f"Loaded FAISS: {self.faiss.ntotal:,} vectors")
        else:
            logger.warning(f"FAISS index not found: {FAISS_INDEX_FILE}")

        if BM25_FILE.exists():
            with open(BM25_FILE, "rb") as f:
                self.bm25 = pickle.load(f)
            logger.info("Loaded BM25 index")
        else:
            logger.warning(f"BM25 index not found: {BM25_FILE}")

        if METADATA_DB_FILE.exists():
            self.db_conn = sqlite3.connect(str(METADATA_DB_FILE), check_same_thread=False)
            self.db_conn.row_factory = sqlite3.Row
            cursor = self.db_conn.cursor()

            # Detect table name
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            if tables:
                self.table_name = tables[0][0]

                # Detect ID column
                cursor.execute(f"PRAGMA table_info({self.table_name});")
                cols = [c[1] for c in cursor.fetchall()]
                logger.info(f"Database table: {self.table_name} | Columns: {cols}")

                for candidate in ("chunk_id", "id", "pk", "index"):
                    if candidate in cols:
                        self.id_column = candidate
                        break
                else:
                    self.id_column = "rowid"
                    self.use_rowid = True
                    logger.warning("No ID column found — falling back to rowid.")

                # Validate alignment with FAISS
                if self.faiss is not None:
                    cursor.execute(f"SELECT count(*) FROM {self.table_name}")
                    count = cursor.fetchone()[0]
                    logger.info(f"Database row count: {count:,}")
                    if count != self.faiss.ntotal:
                        logger.warning(
                            f"Alignment mismatch: FAISS={self.faiss.ntotal}, DB={count}. "
                            "Using rowid fallback."
                        )
                        self.id_column = "rowid"
                        self.use_rowid = True
        else:
            logger.warning(f"Metadata DB not found: {METADATA_DB_FILE}")

    def get_chunk_by_id(self, idx: int) -> dict:
        """
        Fetch chunk text and metadata by FAISS index position (0-based).
        Falls back to rowid (SQLite 1-based) if exact-ID lookup fails.
        """
        if not self.db_conn:
            return {}
        cursor = self.db_conn.cursor()
        try:
            # Primary: try matching the stored integer/text ID
            cursor.execute(
                f"SELECT * FROM {self.table_name} WHERE {self.id_column} = ?", (idx,)
            )
            row = cursor.fetchone()
            if row:
                return dict(row)

            # Fallback: physical row position (FAISS 0-based → rowid 1-based)
            cursor.execute(
                f"SELECT * FROM {self.table_name} WHERE rowid = ?", (idx + 1,)
            )
            row = cursor.fetchone()
            return dict(row) if row else {}
        except Exception as e:
            logger.error(f"Error fetching chunk idx={idx}: {e}")
            return {}

    def get_parent_text(self, parent_id: str) -> str:
        """Fetch parent chunk text by its UUID parent_id."""
        if not self.db_conn or not parent_id:
            return ""
        cursor = self.db_conn.cursor()
        try:
            cursor.execute(
                f"SELECT text FROM {self.table_name} WHERE {self.id_column} = ?",
                (parent_id,)
            )
            row = cursor.fetchone()
            if row:
                return row["text"]
        except Exception:
            pass
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# Offline Index Builder (called by build_indexes.py and kaggle_build.py)
# ─────────────────────────────────────────────────────────────────────────────
def run_indexing(force_rebuild: bool = False) -> dict:
    """
    Build FAISS + SQLite + BM25 indexes from data/processed/chunks.json.
    Designed to run on Kaggle T4 GPU (embedding step uses GPU automatically).

    Returns dict with {faiss, bm25} objects.
    """
    if not force_rebuild and all(
        f.exists() for f in (FAISS_INDEX_FILE, METADATA_DB_FILE, BM25_FILE)
    ):
        logger.info("All indexes already exist — loading and returning.")
        idx = Indexer()
        idx.load_all()
        return {"faiss": idx.faiss, "bm25": idx.bm25}

    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            f"Chunks file not found: {CHUNKS_FILE}. Run ingestion first."
        )

    logger.info("Starting index build from chunks.json...")
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    parents: list[dict] = data["parents"]
    children: list[dict] = data["children"]
    logger.info(f"Loaded {len(parents):,} parents and {len(children):,} child chunks.")

    # ── 1. Embedding (GPU-accelerated on Kaggle T4) ───────────────────────────
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    texts = [c["text"] for c in children]
    logger.info(f"Embedding {len(texts):,} child chunks (batch_size=256)...")
    embeddings = model.encode(
        texts, batch_size=256, show_progress_bar=True, normalize_embeddings=True
    )
    embeddings = np.array(embeddings).astype(np.float32)

    # ── 2. FAISS Index ────────────────────────────────────────────────────────
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product on L2-normalized = cosine
    index.add(embeddings)
    faiss.write_index(index, str(FAISS_INDEX_FILE))
    logger.info(f"Saved FAISS index ({FAISS_INDEX_FILE.stat().st_size / 1e6:.1f} MB)")

    # ── 3. SQLite Metadata DB ─────────────────────────────────────────────────
    if METADATA_DB_FILE.exists():
        os.remove(METADATA_DB_FILE)

    conn = sqlite3.connect(str(METADATA_DB_FILE))
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE chunks (
            id          TEXT PRIMARY KEY,
            parent_id   TEXT,
            text        TEXT NOT NULL,
            source      TEXT,
            page_number INTEGER,
            synthetic   INTEGER DEFAULT 0
        )
    """)
    # Store ALL chunks (parents + children) so parent text lookup works
    all_chunks = parents + children
    unique_chunks = list({c["id"]: c for c in all_chunks}.values())
    cursor.executemany(
        """
        INSERT INTO chunks (id, parent_id, text, source, page_number, synthetic)
        VALUES (:id, :parent_id, :text, :source, :page_number, :synthetic)
        """,
        unique_chunks,
    )
    conn.commit()
    conn.close()
    logger.info(f"Saved SQLite DB: {METADATA_DB_FILE.stat().st_size / 1e6:.1f} MB, "
                f"{len(unique_chunks):,} rows")

    # ── 4. BM25 Index ─────────────────────────────────────────────────────────
    from rank_bm25 import BM25Okapi
    tokenized_corpus = [t.lower().split() for t in texts]
    bm25 = BM25Okapi(tokenized_corpus)
    with open(BM25_FILE, "wb") as f:
        pickle.dump(bm25, f)
    logger.info(f"Saved BM25 index: {BM25_FILE.stat().st_size / 1e6:.1f} MB")

    logger.info("Index build complete.")
    return {"faiss": index, "bm25": bm25}
