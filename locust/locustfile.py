"""
locust/locustfile.py — Load test script for the live ArXiv RAG API.

Run against the deployed HF Spaces URL (not localhost):
    locust -f locust/locustfile.py --host https://yourname-rag-demo.hf.space

Scenarios to run sequentially:
  1. 10 concurrent users  (ramp-up: 30s)
  2. 25 concurrent users  (ramp-up: 60s)
  3. 50 concurrent users  (ramp-up: 120s)

Note: At 50 concurrent users, Groq's free tier (30 req/min) becomes the bottleneck —
HTTP 429 responses are expected and documented as a production finding.
"""

import csv
import random
from pathlib import Path

from locust import HttpUser, between, task

_TEST_SET_PATH = Path(__file__).parent.parent / "tests" / "test_set.csv"
_FALLBACK_QUESTIONS = [
    "What is retrieval-augmented generation?",
    "How does BERT differ from GPT in pretraining objectives?",
    "What are the main challenges in dense passage retrieval?",
    "Explain the attention mechanism in transformer models.",
    "How does BM25 rank documents compared to neural retrieval?",
    "What is the role of cross-encoders in information retrieval?",
    "How does knowledge distillation work in NLP?",
    "What are the limitations of fine-tuning large language models?",
    "Describe the architecture of a typical RAG pipeline.",
    "What is reciprocal rank fusion and why is it useful?",
    "How do sparse and dense retrievers complement each other?",
    "What is context precision in RAG evaluation?",
]


def _load_questions() -> list[str]:
    if _TEST_SET_PATH.exists():
        with open(_TEST_SET_PATH, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            questions = [row["question"] for row in reader if row.get("question")]
        if questions:
            return questions
    return _FALLBACK_QUESTIONS


_QUESTIONS = _load_questions()


class RAGUser(HttpUser):
    """Simulates a concurrent user querying the RAG API."""

    wait_time = between(1, 3)  # seconds between tasks per user

    @task(10)
    def query_hybrid_rerank(self):
        """Primary task: full hybrid + reranking pipeline (default)."""
        with self.client.post(
            "/query",
            json={"question": random.choice(_QUESTIONS), "top_k": 5},
            catch_response=True,
            name="/query [hybrid+rerank]",
        ) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code == 429:
                # Groq rate limit — expected at high concurrency, documented finding
                response.failure("Groq rate limit (429)")
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(1)
    def health_check(self):
        """Occasional health probe to verify server is alive."""
        self.client.get("/health", name="/health")
