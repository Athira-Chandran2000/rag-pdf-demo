"""
run_experiments.py — Orchestrate the full three-variant experiment suite.

Runs three pipeline configurations, evaluates each with RAGAS, logs to MLflow
(local or DagsHub), and writes experiments/metrics_cache.json for the /metrics endpoint.

Usage (after build_indexes.py has been run):
    python run_experiments.py

Environment:
    GROQ_API_KEY           — required (Groq LLM + RAGAS judge)
    MLFLOW_TRACKING_URI    — optional, DagsHub remote URI
    DAGSHUB_TOKEN          — optional, for DagsHub auth
"""

# Load .env before anything else
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / ".env")

import json
import logging
import os

from app.pipeline import RAGPipeline
from app.evaluator import (
    generate_test_set,
    run_ragas_eval,
    compute_latency_stats,
    log_experiment_to_mlflow,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
METRICS_CACHE_FILE = BASE_DIR / "experiments" / "metrics_cache.json"
METRICS_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Experiment Variants
# ─────────────────────────────────────────────────────────────────────────────
VARIANTS = [
    {
        "name": "1. Baseline (Dense Only)",
        "params": {
            "embedding_model": "all-MiniLM-L6-v2",
            "chunk_size_parent": 512,
            "chunk_size_child": 128,
            "retrieval_k": 20,
            "reranker": "none",
            "llm": "llama-3.1-8b-instant",
            "hybrid": False,
        },
        "pipeline_kwargs": {
            "retrieval_top_k": 20,
            "rerank_top_n": 5,
            "dense_only": True,
            "skip_rerank": True,
        },
    },
    {
        "name": "2. Dense + Reranker",
        "params": {
            "embedding_model": "all-MiniLM-L6-v2",
            "chunk_size_parent": 512,
            "chunk_size_child": 128,
            "retrieval_k": 20,
            "reranker": "ms-marco-MiniLM-L-12-v2",
            "llm": "llama-3.1-8b-instant",
            "hybrid": False,
        },
        "pipeline_kwargs": {
            "retrieval_top_k": 20,
            "rerank_top_n": 5,
            "dense_only": True,
            "skip_rerank": False,
        },
    },
    {
        "name": "3. Hybrid + Reranker (Full)",
        "params": {
            "embedding_model": "all-MiniLM-L6-v2",
            "chunk_size_parent": 512,
            "chunk_size_child": 128,
            "retrieval_k": 20,
            "reranker": "ms-marco-MiniLM-L-12-v2",
            "llm": "llama-3.1-8b-instant",
            "hybrid": True,
        },
        "pipeline_kwargs": {
            "retrieval_top_k": 20,
            "rerank_top_n": 5,
            "dense_only": False,
            "skip_rerank": False,
        },
    },
]


def main():
    # ── Step 1: Generate test set ─────────────────────────────────────────────
    logger.info("=== Step 1: Generating synthetic test set ===")
    test_pairs = generate_test_set(n_pairs=100)
    logger.info(f"Test set ready: {len(test_pairs)} QA pairs")

    metrics_cache = {"variants": []}

    # ── Step 2: Run each variant ──────────────────────────────────────────────
    for variant in VARIANTS:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running variant: {variant['name']}")
        logger.info(f"{'='*60}")

        pipeline = RAGPipeline(**variant["pipeline_kwargs"])

        # RAGAS evaluation (50 questions for speed; set to 100 for full eval)
        ragas_scores = run_ragas_eval(
            pipeline=pipeline,
            variant_name=variant["name"],
            max_questions=50,
        )

        # Per-question latency log (all 50 questions)
        latency_records = []
        for pair in test_pairs[:50]:
            try:
                result = pipeline.query(pair["question"])
                latency_records.append(result["latencies"])
            except Exception as e:
                logger.warning(f"Latency eval error: {e}")

        latency_stats = compute_latency_stats(latency_records)

        # MLflow logging (local or DagsHub remote)
        run_id = log_experiment_to_mlflow(
            variant_name=variant["name"],
            params=variant["params"],
            ragas_scores=ragas_scores,
            latency_stats=latency_stats,
        )

        # Cache for /metrics and /experiments endpoints
        entry = {
            "name": variant["name"],
            "mlflow_run_id": run_id,
            **{f"ragas_{k}": round(v, 4) for k, v in ragas_scores.items()},
            "latency_p50_ms": latency_stats.get("total_ms", {}).get("p50"),
            "latency_p90_ms": latency_stats.get("total_ms", {}).get("p90"),
            "latency_p99_ms": latency_stats.get("total_ms", {}).get("p99"),
            "latency_details": latency_stats,
        }
        metrics_cache["variants"].append(entry)

        logger.info(f"Variant '{variant['name']}' complete:")
        logger.info(f"  RAGAS: {ragas_scores}")
        logger.info(f"  P99 latency: {entry.get('latency_p99_ms')} ms")

    # ── Step 3: Write metrics cache ───────────────────────────────────────────
    with open(METRICS_CACHE_FILE, "w") as f:
        json.dump(metrics_cache, f, indent=2)
    logger.info(f"\nMetrics cache saved → {METRICS_CACHE_FILE}")
    logger.info("=== All experiments complete ===")


if __name__ == "__main__":
    main()
