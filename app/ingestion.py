"""
ingestion.py — PDF parsing, hierarchical chunking, ArXiv download, synthetic expansion.

Workflow:
  1. Download ArXiv PDFs via arxiv API (polite rate-limited).
  2. Parse each PDF with PyMuPDF; strip headers/footers by position heuristics.
  3. Produce parent chunks (512 tok) and child chunks (128 tok) with lineage.
  4. Optionally expand to ~50k chunks via fast local text augmentation (no API calls).
  5. Persist everything as JSON — subsequent runs skip reprocessing.
"""

import os
import re
import json
import uuid
import time
import random
import logging
import hashlib
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
import arxiv
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
PDF_DIR = BASE_DIR / "data" / "pdfs"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
CHUNKS_FILE = PROCESSED_DIR / "chunks.json"

PDF_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# ArXiv Download
# ─────────────────────────────────────────────────────────────────────────────
ARXIV_QUERIES = [
    "large language models retrieval augmented generation",
    "transformer neural network attention mechanism",
    "BERT GPT language model pretraining",
    "knowledge graph embedding representation learning",
    "natural language processing text classification",
    "question answering machine reading comprehension",
    "semantic search dense retrieval embedding",
    "document summarization abstractive extractive",
    "information retrieval ranking neural",
    "multimodal learning vision language model",
]


def download_arxiv_papers(
    target_count: int = 30,
    pdf_dir: Path = PDF_DIR,
    delay: float = 3.0,
) -> list[Path]:
    """
    Download up to `target_count` ArXiv PDFs across multiple queries.
    Politely rate-limited. Returns list of downloaded PDF paths.
    """
    pdf_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []
    per_query = max(1, target_count // len(ARXIV_QUERIES) + 1)

    for query in ARXIV_QUERIES:
        if len(downloaded) >= target_count:
            break
        logger.info(f"ArXiv search: '{query}' (max {per_query} results)")
        try:
            search = arxiv.Search(
                query=query,
                max_results=per_query,
                sort_by=arxiv.SortCriterion.Relevance,
            )
            for result in search.results():
                if len(downloaded) >= target_count:
                    break
                arxiv_id = result.entry_id.split("/")[-1].replace("/", "_")
                dest = pdf_dir / f"{arxiv_id}.pdf"
                if dest.exists():
                    logger.info(f"  Already downloaded: {dest.name}")
                    downloaded.append(dest)
                    continue
                try:
                    result.download_pdf(dirpath=str(pdf_dir), filename=dest.name)
                    logger.info(f"  Downloaded: {dest.name}")
                    downloaded.append(dest)
                    time.sleep(delay)
                except Exception as e:
                    logger.warning(f"  Failed to download {arxiv_id}: {e}")
        except Exception as e:
            logger.warning(f"ArXiv search failed for query '{query}': {e}")
        time.sleep(delay)

    logger.info(f"Total PDFs: {len(downloaded)}")
    return downloaded[:target_count]


# ─────────────────────────────────────────────────────────────────────────────
# PDF Text Extraction
# ─────────────────────────────────────────────────────────────────────────────
def _clean_text(text: str) -> str:
    """Remove excessive whitespace and normalize unicode dashes."""
    text = re.sub(r"\s+", " ", text)
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    return text.strip()


def _is_header_footer(block, page_height: float, threshold: float = 0.05) -> bool:
    """True if the text block sits in the top or bottom 5% of the page."""
    _, y0, _, y1 = block[:4]
    top_band = page_height * threshold
    bot_band = page_height * (1.0 - threshold)
    return y1 < top_band or y0 > bot_band


def extract_pages(pdf_path: Path) -> list[dict]:
    """
    Extract text page-by-page from a PDF, stripping header/footer bands.
    Returns list of {page_number, text, source}.
    """
    pages = []
    try:
        doc = fitz.open(str(pdf_path))
        for page_num, page in enumerate(doc, start=1):
            page_height = page.rect.height
            blocks = page.get_text("blocks")
            body_blocks = [
                b for b in blocks
                if not _is_header_footer(b, page_height)
            ]
            text = " ".join(b[4] for b in body_blocks if isinstance(b[4], str))
            text = _clean_text(text)

            # Fallback for suspiciously short pages (possible scan)
            if len(text) < 50:
                raw = _clean_text(page.get_text("text"))
                if len(raw) > len(text):
                    text = raw
                logger.debug(f"  Page {page_num} of {pdf_path.name}: short text ({len(text)} chars)")

            pages.append({
                "page_number": page_num,
                "text": text,
                "source": pdf_path.name,
            })
        doc.close()
    except Exception as e:
        logger.error(f"Failed to extract {pdf_path.name}: {e}")
    return pages


# ─────────────────────────────────────────────────────────────────────────────
# Hierarchical Chunking
# ─────────────────────────────────────────────────────────────────────────────
_PARENT_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=512 * 4,       # ~512 tokens (4 chars/token estimate)
    chunk_overlap=50 * 4,     # ~50 token overlap
    separators=["\n\n", "\n", ". ", " ", ""],
)
_CHILD_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=128 * 4,       # ~128 tokens
    chunk_overlap=20 * 4,     # ~20 token overlap
    separators=["\n\n", "\n", ". ", " ", ""],
)


def chunk_pages(pages: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Given extracted pages, produce parent and child chunks.

    Returns:
        (parent_chunks, child_chunks)
        Each chunk: {id, parent_id, text, source, page_number, synthetic}
    """
    full_text = "\n\n".join(p["text"] for p in pages if p["text"])
    if not full_text.strip():
        return [], []

    source = pages[0]["source"] if pages else "unknown"

    # Page-lookup for assigning page numbers to chunks by character offset
    page_boundaries: list[tuple[int, int]] = []
    offset = 0
    for p in pages:
        page_boundaries.append((offset, p["page_number"]))
        offset += len(p["text"]) + 2  # +2 for "\n\n"

    def page_of(char_offset: int) -> int:
        pn = 1
        for start, num in page_boundaries:
            if char_offset >= start:
                pn = num
            else:
                break
        return pn

    # Parent chunks
    parent_chunks: list[dict] = []
    parent_texts = _PARENT_SPLITTER.split_text(full_text)
    char_cursor = 0
    for ptxt in parent_texts:
        idx = full_text.find(ptxt, char_cursor)
        if idx == -1:
            idx = char_cursor
        pid = str(uuid.uuid4())
        parent_chunks.append({
            "id": pid,
            "parent_id": None,
            "text": ptxt,
            "source": source,
            "page_number": page_of(idx),
            "synthetic": False,
        })
        char_cursor = max(char_cursor, idx + 1)

    # Child chunks
    child_chunks: list[dict] = []
    for pc in parent_chunks:
        child_texts = _CHILD_SPLITTER.split_text(pc["text"])
        for ctxt in child_texts:
            child_chunks.append({
                "id": str(uuid.uuid4()),
                "parent_id": pc["id"],
                "text": ctxt,
                "source": source,
                "page_number": pc["page_number"],
                "synthetic": False,
            })

    return parent_chunks, child_chunks


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic Expansion — Fast Local Augmentation (zero API calls)
# ─────────────────────────────────────────────────────────────────────────────
def _augment_text(text: str, rng: random.Random, variant: int) -> str:
    """
    Produce a lightly varied version of `text` via deterministic local transforms.

    Strategies (cycled by variant index):
      0 — Shuffle sentences within the chunk
      1 — Drop the first sentence (simulates mid-passage start)
      2 — Drop the last sentence (simulates early cutoff)
      3 — Swap adjacent sentence pairs
      4 — Duplicate the longest sentence as a summary prefix
    """
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if len(sentences) < 2:
        return text

    v = variant % 5
    if v == 0:
        rng.shuffle(sentences)
    elif v == 1:
        sentences = sentences[1:]
    elif v == 2:
        sentences = sentences[:-1]
    elif v == 3:
        for i in range(0, len(sentences) - 1, 2):
            sentences[i], sentences[i + 1] = sentences[i + 1], sentences[i]
    elif v == 4:
        longest = max(sentences, key=len)
        sentences = [longest] + sentences

    return " ".join(sentences)


def expand_chunks_locally(
    child_chunks: list[dict],
    target_total: int = 50_000,
    seed: int = 42,
) -> list[dict]:
    """
    Expand child_chunks to `target_total` via local text augmentation.
    Runs in O(n) time with zero API calls.
    """
    rng = random.Random(seed)
    current_count = len(child_chunks)
    if current_count >= target_total:
        logger.info("Already have enough chunks — skipping expansion.")
        return child_chunks

    needed = target_total - current_count
    logger.info(
        f"Local expansion: {current_count:,} → {target_total:,} chunks "
        f"({needed:,} synthetic, no API calls)"
    )

    real_pool = [c for c in child_chunks if not c.get("synthetic", False)]
    if not real_pool:
        real_pool = child_chunks

    synthetic: list[dict] = []
    variant = 0
    pool_idx = 0

    while len(synthetic) < needed:
        orig = real_pool[pool_idx % len(real_pool)]
        augmented_text = _augment_text(orig["text"], rng, variant)

        if augmented_text and augmented_text != orig["text"]:
            synthetic.append({
                "id": str(uuid.uuid4()),
                "parent_id": orig["parent_id"],
                "text": augmented_text,
                "source": orig["source"],
                "page_number": orig["page_number"],
                "synthetic": True,
            })

        pool_idx += 1
        if pool_idx % len(real_pool) == 0:
            variant += 1
            rng.shuffle(real_pool)

    logger.info(f"Generated {len(synthetic):,} synthetic chunks in O(n) time.")
    return child_chunks + synthetic


# ─────────────────────────────────────────────────────────────────────────────
# Top-level Orchestration
# ─────────────────────────────────────────────────────────────────────────────
def run_ingestion(
    pdf_dir: Path = PDF_DIR,
    chunks_file: Path = CHUNKS_FILE,
    target_arxiv_papers: int = 30,
    target_total_chunks: int = 50_000,
    skip_expansion: bool = False,
    force_reprocess: bool = False,
) -> tuple[list[dict], list[dict]]:
    """
    Full ingestion pipeline. Returns (parent_chunks, child_chunks).
    If chunks_file already exists and force_reprocess is False, loads from disk.
    """
    if chunks_file.exists() and not force_reprocess:
        logger.info(f"Loading chunks from cache: {chunks_file}")
        with open(chunks_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data["parents"], data["children"]

    # Step 1: ensure we have PDFs
    pdf_paths = sorted(pdf_dir.glob("*.pdf"))
    if len(pdf_paths) < 5:
        logger.info("Fewer than 5 PDFs found — downloading from ArXiv.")
        pdf_paths = download_arxiv_papers(
            target_count=target_arxiv_papers,
            pdf_dir=pdf_dir,
        )
    else:
        logger.info(f"Found {len(pdf_paths)} existing PDFs — skipping download.")

    # Step 2: parse + chunk
    all_parents: list[dict] = []
    all_children: list[dict] = []
    for pdf_path in pdf_paths:
        logger.info(f"Processing: {pdf_path.name}")
        pages = extract_pages(pdf_path)
        if not pages:
            continue
        parents, children = chunk_pages(pages)
        all_parents.extend(parents)
        all_children.extend(children)
        logger.info(f"  → {len(parents)} parent chunks, {len(children)} child chunks")

    logger.info(f"Total after real PDFs: {len(all_parents)} parents, {len(all_children)} children")

    # Step 3: synthetic expansion
    if not skip_expansion and len(all_children) < target_total_chunks:
        all_children = expand_chunks_locally(all_children, target_total=target_total_chunks)

    # Step 4: persist
    with open(chunks_file, "w", encoding="utf-8") as f:
        json.dump({"parents": all_parents, "children": all_children}, f)
    logger.info(f"Saved {len(all_parents)} parents + {len(all_children)} children → {chunks_file}")

    return all_parents, all_children


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parents, children = run_ingestion(
        target_arxiv_papers=500,
        target_total_chunks=40000,
        skip_expansion=False,
    )
    print(f"Parents: {len(parents)}  Children: {len(children)}")
