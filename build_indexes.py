"""
build_indexes.py — Offline index builder. Run this ONCE before deploying.

Steps:
  1. Download ArXiv PDFs (30 papers by default, 500 on Kaggle)
  2. Parse, clean, and hierarchically chunk all PDFs
  3. Optionally expand to 50k chunks via local augmentation (no API calls)
  4. Embed child chunks with all-MiniLM-L6-v2 (GPU-accelerated on Kaggle T4)
  5. Build and save FAISS index, SQLite metadata DB, BM25 index

After this script finishes, commit the data/processed/ directory to Git LFS.

Usage:
    # Quick test (30 papers, no expansion):
    python build_indexes.py --skip-expansion

    # Full pipeline (500 papers, synthetic expansion to 50k chunks):
    python build_indexes.py --papers 500 --target-chunks 50000

    # Force rebuild even if indexes exist:
    python build_indexes.py --force

    # Use existing PDFs (skip download):
    python build_indexes.py --skip-download --skip-expansion
"""

# Load .env before anything else
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / ".env")

import argparse
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Build RAG indexes from ArXiv PDFs")
    parser.add_argument("--papers", type=int, default=30, help="Number of ArXiv papers to download")
    parser.add_argument("--target-chunks", type=int, default=50_000, help="Target total child chunks (with expansion)")
    parser.add_argument("--skip-expansion", action="store_true", help="Skip synthetic expansion")
    parser.add_argument("--force", action="store_true", help="Force rebuild even if indexes exist")
    parser.add_argument("--skip-download", action="store_true", help="Skip ArXiv download (use existing PDFs)")
    args = parser.parse_args()

    if not os.environ.get("GROQ_API_KEY"):
        logger.warning(
            "GROQ_API_KEY not set. "
            "Synthetic expansion uses local augmentation (no API needed). "
            "Groq key is only required for run_experiments.py and run_experiments.py."
        )

    logger.info("=" * 60)
    logger.info("ArXiv RAG — Offline Index Builder")
    logger.info("=" * 60)
    logger.info(f"Papers to download : {args.papers}")
    logger.info(f"Target chunks      : {args.target_chunks:,}")
    logger.info(f"Skip expansion     : {args.skip_expansion}")
    logger.info(f"Force rebuild      : {args.force}")
    logger.info(f"Skip download      : {args.skip_download}")
    logger.info("=" * 60)

    # ── Step 1+2+3: Ingestion ─────────────────────────────────────────────────
    from app.ingestion import run_ingestion, PDF_DIR, CHUNKS_FILE

    parents, children = run_ingestion(
        pdf_dir=PDF_DIR,
        chunks_file=CHUNKS_FILE,
        target_arxiv_papers=0 if args.skip_download else args.papers,
        target_total_chunks=args.target_chunks,
        skip_expansion=args.skip_expansion,
        force_reprocess=args.force,
    )
    logger.info(f"Ingestion done: {len(parents):,} parent chunks, {len(children):,} child chunks")

    # ── Step 4+5: Indexing ────────────────────────────────────────────────────
    from app.indexer import run_indexing, FAISS_INDEX_FILE, METADATA_DB_FILE, BM25_FILE

    artifacts = run_indexing(force_rebuild=args.force)

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("INDEX BUILD COMPLETE")
    logger.info("=" * 60)
    logger.info(f"FAISS vectors  : {artifacts['faiss'].ntotal:,}")
    for label, path in [("FAISS file", FAISS_INDEX_FILE), ("SQLite DB", METADATA_DB_FILE), ("BM25 index", BM25_FILE)]:
        if path.exists():
            logger.info(f"{label:<14}: {path}  ({path.stat().st_size / 1e6:.1f} MB)")

    logger.info("\nNext steps:")
    logger.info("  1. git lfs install")
    logger.info("  2. git add data/processed/ && git commit -m 'Add processed indexes'")
    logger.info("  3. git push → HF Spaces auto-deploys")
    logger.info("  4. python run_experiments.py  (after deployment)")


if __name__ == "__main__":
    main()
