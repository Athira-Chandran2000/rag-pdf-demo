"""
kaggle_build.py — Docker CMD entry point for Hugging Face Spaces.

On Dockerfile CMD: runs ingestion + indexing ONLY if indexes are absent.
This avoids re-downloading 500 papers on every container restart.

When running on HF Spaces, GROQ_API_KEY is injected as a Space secret.
"""

import logging
import os
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("kaggle_build")

BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
FAISS_INDEX_FILE = PROCESSED_DIR / "faiss.index"
METADATA_DB_FILE = PROCESSED_DIR / "metadata.db"
BM25_FILE = PROCESSED_DIR / "bm25.pkl"
CHUNKS_FILE = PROCESSED_DIR / "chunks.json"


def indexes_exist() -> bool:
    return all(f.exists() for f in (FAISS_INDEX_FILE, METADATA_DB_FILE, BM25_FILE))


def main():
    logger.info("=" * 60)
    logger.info("Kaggle Build / HF Spaces Startup Check")
    logger.info("=" * 60)

    if indexes_exist():
        logger.info("All indexes present — skipping ingestion and indexing.")
        logger.info(f"  FAISS : {FAISS_INDEX_FILE.stat().st_size / 1e6:.1f} MB")
        logger.info(f"  SQLite: {METADATA_DB_FILE.stat().st_size / 1e6:.1f} MB")
        logger.info(f"  BM25  : {BM25_FILE.stat().st_size / 1e6:.1f} MB")
        return

    logger.info("Indexes not found — running ingestion pipeline...")

    # Step 1: Ingestion (downloads ArXiv papers + chunking)
    from app.ingestion import run_ingestion
    parents, children = run_ingestion(
        target_arxiv_papers=500,
        target_total_chunks=40_000,
        skip_expansion=False,
        force_reprocess=False,
    )
    logger.info(f"Generated {len(parents):,} parents and {len(children):,} child chunks.")

    # Step 2: Build indexes (FAISS + SQLite + BM25)
    from app.indexer import run_indexing
    artifacts = run_indexing(force_rebuild=True)

    logger.info("=" * 60)
    logger.info("BUILD COMPLETE")
    logger.info(f"  FAISS vectors: {artifacts['faiss'].ntotal:,}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
