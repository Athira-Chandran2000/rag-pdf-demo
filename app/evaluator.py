"""
evaluator.py — Synthetic test set generation + RAGAS evaluation + latency statistics + MLflow logging.

Workflow:
  1. generate_test_set()        → tests/test_set.csv  (100 QA pairs via Groq)
  2. run_ragas_eval()           → RAGAS metrics dict; saves per-question CSV
  3. compute_latency_stats()    → p50/p90/p99 per stage
  4. log_experiment_to_mlflow() → logs params, metrics, artifacts to MLflow (local or DagsHub)

RAGAS uses Groq llama-3.1-8b-instant as the judge LLM via LangChain's ChatGroq wrapper.
Run on Kaggle T4 GPU for faster RAGAS judge LLM calls.
"""

import csv
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np
from groq import Groq

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
CHUNKS_FILE = BASE_DIR / "data" / "processed" / "chunks.json"
TESTS_DIR = BASE_DIR / "tests"
TEST_SET_CSV = TESTS_DIR / "test_set.csv"
TESTS_DIR.mkdir(parents=True, exist_ok=True)

EVAL_RESULTS_DIR = BASE_DIR / "experiments" / "eval_results"
EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

QA_GEN_PROMPT = """\
Given the following passage from an academic paper, generate exactly ONE specific factual question \
that can be answered directly from this passage, and provide the reference answer.

Format your response EXACTLY as:
Question: <question>
Answer: <answer>

Passage:
{passage}"""


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic Test Set Generation
# ─────────────────────────────────────────────────────────────────────────────
def generate_test_set(
    n_pairs: int = 100,
    out_csv: Path = TEST_SET_CSV,
    force_regenerate: bool = False,
) -> list[dict]:
    """
    Generate `n_pairs` QA pairs by prompting Groq on sampled child chunks.
    Saves to `out_csv`. Returns list of {question, reference_answer, source_chunk_id}.
    """
    if out_csv.exists() and not force_regenerate:
        logger.info(f"Test set already exists: {out_csv}")
        with open(out_csv, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(f"Chunks file not found: {CHUNKS_FILE}. Run ingestion first.")

    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    children: list[dict] = data["children"]

    # Prefer real (non-synthetic) chunks
    real_chunks = [c for c in children if not c.get("synthetic", False)]
    pool = real_chunks if len(real_chunks) >= n_pairs else children
    sampled = random.sample(pool, min(n_pairs, len(pool)))

    pairs: list[dict] = []
    for chunk in sampled:
        if len(pairs) >= n_pairs:
            break
        prompt = QA_GEN_PROMPT.format(passage=chunk["text"][:1200])
        try:
            resp = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=256,
                temperature=0.4,
            )
            content = resp.choices[0].message.content.strip()
            q_lines = [l for l in content.split("\n") if l.lower().startswith("question:")]
            a_lines = [l for l in content.split("\n") if l.lower().startswith("answer:")]
            if q_lines and a_lines:
                pairs.append({
                    "question": q_lines[0].split(":", 1)[1].strip(),
                    "reference_answer": a_lines[0].split(":", 1)[1].strip(),
                    "source_chunk_id": chunk["id"],
                })
        except Exception as e:
            logger.warning(f"QA gen failed for chunk {chunk['id']}: {e}")
        time.sleep(1.5)  # respect Groq free-tier rate limit

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "reference_answer", "source_chunk_id"])
        writer.writeheader()
        writer.writerows(pairs)

    logger.info(f"Test set saved: {len(pairs)} pairs → {out_csv}")
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# RAGAS Evaluation
# ─────────────────────────────────────────────────────────────────────────────
def run_ragas_eval(
    pipeline,
    variant_name: str = "hybrid_rerank",
    test_csv: Path = TEST_SET_CSV,
    max_questions: int = 50,
) -> dict:
    """
    Evaluate `pipeline` on the test set using RAGAS (Groq as judge LLM).

    Metrics computed:
      - faithfulness       (claims grounded in retrieved context)
      - answer_relevancy   (does the answer address the question)
      - context_precision  (fraction of retrieved chunks that were useful)
      - context_recall     (whether retrieval captured the needed info)

    Saves per-question CSV to experiments/eval_results/<variant_name>.csv.
    Returns aggregate metric dict.
    """
    try:
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )
        from datasets import Dataset
        from langchain_groq import ChatGroq
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except ImportError as e:
        logger.error(f"RAGAS / LangChain dependencies missing: {e}")
        return {}

    if not test_csv.exists():
        logger.error("Test set CSV not found. Run generate_test_set() first.")
        return {}

    with open(test_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))[:max_questions]

    questions, ground_truths, answers, contexts = [], [], [], []
    latency_records: list[dict] = []

    for row in rows:
        try:
            result = pipeline.query(row["question"])
            questions.append(row["question"])
            ground_truths.append(row["reference_answer"])
            answers.append(result["answer"])
            contexts.append([c.get("text", "") for c in result["chunks_used"]])
            latency_records.append(result["latencies"])
        except Exception as e:
            logger.warning(f"Pipeline error on '{row['question']}': {e}")

    if not questions:
        logger.error("No questions evaluated — check pipeline and test CSV.")
        return {}

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    llm = ChatGroq(
        model_name="llama-3.1-8b-instant",
        groq_api_key=os.environ["GROQ_API_KEY"],
        temperature=0,
    )
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    logger.info(f"Running RAGAS evaluation for variant: {variant_name}")
    try:
        results = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=llm,
            embeddings=embeddings,
        )
        df = results.to_pandas()
        aggregate = {
            col: float(np.mean(df[col].dropna()))
            for col in ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
            if col in df.columns
        }
    except Exception as e:
        logger.error(f"RAGAS evaluation failed: {e}")
        aggregate = {}

    # Save per-question CSV
    out_csv = EVAL_RESULTS_DIR / f"{variant_name.replace(' ', '_').replace('.', '')}.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["question", "answer", "ground_truth", "latency_total_ms"])
        for i, q in enumerate(questions):
            lat = latency_records[i].get("total_ms", 0) if i < len(latency_records) else 0
            writer.writerow([q, answers[i], ground_truths[i], f"{lat:.1f}"])

    logger.info(f"RAGAS results for {variant_name}: {aggregate}")
    logger.info(f"Saved per-question CSV → {out_csv}")
    return aggregate


# ─────────────────────────────────────────────────────────────────────────────
# Latency Statistics
# ─────────────────────────────────────────────────────────────────────────────
def compute_latency_stats(latency_records: list[dict]) -> dict:
    """
    Compute p50/p90/p99 per latency stage from a list of latency dicts.

    Returns:
        { stage_key: {p50: float, p90: float, p99: float} }
    """
    if not latency_records:
        return {}
    stages = list(latency_records[0].keys())
    stats: dict[str, dict] = {}
    for stage in stages:
        values = [r[stage] for r in latency_records if stage in r and r[stage] is not None]
        if values:
            stats[stage] = {
                "p50": float(np.percentile(values, 50)),
                "p90": float(np.percentile(values, 90)),
                "p99": float(np.percentile(values, 99)),
            }
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# MLflow Experiment Logging
# ─────────────────────────────────────────────────────────────────────────────
def log_experiment_to_mlflow(
    variant_name: str,
    params: dict,
    ragas_scores: dict,
    latency_stats: dict,
    tracking_uri: Optional[str] = None,
) -> str:
    """
    Log a RAG experiment run to MLflow.

    If MLFLOW_TRACKING_URI env var is set (e.g. DagsHub URI), logs remotely.
    Otherwise logs locally to ./experiments/mlruns.

    Returns the MLflow run ID string.
    """
    try:
        import mlflow
    except ImportError:
        logger.error("mlflow not installed — skipping experiment logging.")
        return ""

    uri = (
        tracking_uri
        or os.environ.get("MLFLOW_TRACKING_URI")
        or str(BASE_DIR / "experiments" / "mlruns")
    )
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("rag-pdf-arxiv")

    with mlflow.start_run(run_name=variant_name) as run:
        mlflow.log_params(params)

        for metric, value in ragas_scores.items():
            if value is not None:
                mlflow.log_metric(f"ragas_{metric}", float(value))

        for stage, pcts in latency_stats.items():
            for pct_name, val in pcts.items():
                mlflow.log_metric(f"latency_{stage}_{pct_name}", float(val))

        if TEST_SET_CSV.exists():
            mlflow.log_artifact(str(TEST_SET_CSV), artifact_path="test_data")

        return run.info.run_id

    return ""
