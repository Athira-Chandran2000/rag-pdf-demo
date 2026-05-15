"""
main.py — FastAPI application: Chat UI, /query, /health, /metrics, /experiments endpoints.

Design:
- All heavy objects (FAISS, SQLite, BM25, models) load ONCE at startup via lifespan.
- Duplicate @app.get("/") route from old code is removed — single clean root handler.
- /metrics reads from experiments/metrics_cache.json (written by run_experiments.py).
- The Chat UI includes per-stage latency breakdown in every response bubble.
"""

from pathlib import Path as _Path
from dotenv import load_dotenv as _load_dotenv
_load_dotenv(_Path(__file__).resolve().parent.parent / ".env")

import json
import logging
import os
import time
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from app.pipeline import RAGPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
METRICS_CACHE = BASE_DIR / "experiments" / "metrics_cache.json"


class AppState:
    pipeline: Optional[RAGPipeline] = None
    startup_time: float = 0.0
    ready: bool = False


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.perf_counter()
    try:
        logger.info("Loading RAGPipeline at startup...")
        state.pipeline = RAGPipeline()
        state.startup_time = (time.perf_counter() - t0) * 1000
        state.ready = True
        logger.info(f"Pipeline ready in {state.startup_time:.0f}ms")
    except Exception as e:
        logger.error(f"Startup error: {e}", exc_info=True)
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="ArXiv Production RAG",
    description=(
        "Hybrid RAG pipeline (FAISS + BM25 + Flashrank) over 500 ArXiv ML papers. "
        "Powered by Groq llama-3.1-8b-instant."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Chat UI
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>ArXiv RAG — Research Assistant</title>
        <meta name="description" content="Production RAG pipeline over 500 ArXiv ML papers. Hybrid retrieval, cross-encoder reranking, Groq LLM.">
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
        <style>
            :root {
                --primary: #6366f1;
                --primary-light: #818cf8;
                --bg: #0f172a;
                --card: #1e293b;
                --border: #334155;
                --text: #f8fafc;
                --muted: #94a3b8;
                --green: #4ade80;
                --green-dark: #064e3b;
            }
            * { box-sizing: border-box; margin: 0; padding: 0; }
            body { font-family: 'Inter', sans-serif; background: var(--bg); color: var(--text); display: flex; height: 100vh; overflow: hidden; }

            /* Sidebar */
            .sidebar {
                width: 280px; min-width: 280px;
                background: var(--card);
                border-right: 1px solid var(--border);
                padding: 1.5rem;
                display: flex; flex-direction: column; gap: 0.5rem;
                overflow-y: auto;
            }
            .badge {
                display: inline-flex; align-items: center; gap: 0.4rem;
                background: var(--green); color: var(--green-dark);
                padding: 0.3rem 0.8rem; border-radius: 20px;
                font-size: 0.72rem; font-weight: 700; margin-bottom: 1rem;
            }
            .badge::before { content: "●"; font-size: 0.6rem; }
            .sidebar h2 { font-size: 1.1rem; font-weight: 700; margin-bottom: 1rem; color: var(--text); }
            .stat-group { margin-bottom: 1rem; }
            .stat-label { font-size: 0.7rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em; }
            .stat-value { font-size: 0.95rem; font-weight: 600; margin-top: 0.2rem; }
            .divider { border: 0; border-top: 1px solid var(--border); margin: 1rem 0; }
            .nav-link {
                display: flex; align-items: center; gap: 0.5rem;
                color: var(--primary-light); text-decoration: none;
                font-size: 0.85rem; padding: 0.4rem 0.5rem; border-radius: 6px;
                transition: background 0.15s;
            }
            .nav-link:hover { background: rgba(99, 102, 241, 0.1); }
            .pipeline-step { font-size: 0.75rem; color: var(--muted); margin: 0.2rem 0; padding-left: 0.5rem; }

            /* Main */
            .main { flex: 1; display: flex; flex-direction: column; min-width: 0; }
            .messages {
                flex: 1; overflow-y: auto; padding: 2rem;
                display: flex; flex-direction: column; gap: 1.5rem;
                scroll-behavior: smooth;
            }
            .message { padding: 1.1rem 1.3rem; border-radius: 14px; max-width: 82%; line-height: 1.65; font-size: 0.93rem; }
            .user { align-self: flex-end; background: var(--primary); box-shadow: 0 4px 14px rgba(99,102,241,0.35); }
            .ai   { align-self: flex-start; background: var(--card); border: 1px solid var(--border); }
            .thinking { align-self: flex-start; background: var(--card); border: 1px solid var(--border); color: var(--muted); font-style: italic; font-size: 0.85rem; padding: 0.7rem 1rem; border-radius: 10px; }

            .citations { font-size: 0.78rem; color: var(--primary-light); margin-top: 0.8rem; border-top: 1px solid var(--border); padding-top: 0.7rem; }
            .latency-row { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-top: 0.7rem; }
            .latency-pill { font-size: 0.68rem; background: var(--bg); border: 1px solid var(--border); padding: 0.2rem 0.5rem; border-radius: 4px; color: var(--muted); }
            .latency-total { color: var(--green); border-color: var(--green); }

            /* Input */
            .input-area {
                padding: 1.2rem 2rem;
                background: var(--card);
                border-top: 1px solid var(--border);
                display: flex; gap: 0.8rem; align-items: center;
            }
            #msg {
                flex: 1;
                background: var(--bg); border: 1px solid var(--border); color: var(--text);
                padding: 0.85rem 1rem; border-radius: 10px; font-size: 0.93rem; outline: none;
                transition: border-color 0.2s;
            }
            #msg:focus { border-color: var(--primary); }
            #sendBtn {
                background: var(--primary); color: white; border: none;
                padding: 0.85rem 1.6rem; border-radius: 10px; cursor: pointer;
                font-weight: 600; font-size: 0.93rem;
                transition: opacity 0.2s, transform 0.15s;
                white-space: nowrap;
            }
            #sendBtn:hover:not(:disabled) { opacity: 0.88; transform: translateY(-1px); }
            #sendBtn:disabled { opacity: 0.5; cursor: default; }
        </style>
    </head>
    <body>
        <div class="sidebar">
            <div class="badge">PRODUCTION LIVE</div>
            <h2>RAG Explorer</h2>
            <div class="stat-group">
                <div class="stat-label">Corpus</div>
                <div class="stat-value">500 ArXiv Papers</div>
            </div>
            <div class="stat-group">
                <div class="stat-label">Index Scale</div>
                <div class="stat-value">~35,000 Chunks</div>
            </div>
            <div class="stat-group">
                <div class="stat-label">Retrieval</div>
                <div class="stat-value">FAISS + BM25 + RRF</div>
            </div>
            <div class="stat-group">
                <div class="stat-label">Reranker</div>
                <div class="stat-value">Flashrank (cross-encoder)</div>
            </div>
            <div class="stat-group">
                <div class="stat-label">LLM</div>
                <div class="stat-value">Groq llama-3.1-8b</div>
            </div>
            <hr class="divider">
            <a class="nav-link" href="/docs">📘 Swagger API</a>
            <a class="nav-link" href="/metrics">📊 Metrics</a>
            <a class="nav-link" href="/experiments">🧪 Experiments</a>
            <a class="nav-link" href="/health">💚 Health</a>
            <hr class="divider">
            <div class="stat-label" style="margin-bottom:0.5rem">Pipeline</div>
            <div class="pipeline-step">① Dense (FAISS cosine)</div>
            <div class="pipeline-step">② Sparse (BM25 Okapi)</div>
            <div class="pipeline-step">③ RRF Fusion → top 20</div>
            <div class="pipeline-step">④ Flashrank rerank → top 5</div>
            <div class="pipeline-step">⑤ Parent chunk swap</div>
            <div class="pipeline-step">⑥ Groq LLM generation</div>
        </div>
        <div class="main">
            <div class="messages" id="chat">
                <div class="message ai">
                    Hello! I'm your ArXiv Research Assistant. I've indexed <strong>500 recent ML/NLP/AI papers</strong>.<br><br>
                    Try asking: <em>"How does RAG work?"</em> or <em>"What is the attention mechanism in Transformers?"</em>
                </div>
            </div>
            <div class="input-area">
                <input type="text" id="msg" placeholder="Ask a technical question about AI, NLP, or Machine Learning..." autocomplete="off"
                    onkeypress="if(event.key==='Enter') send()">
                <button id="sendBtn" onclick="send()">Ask</button>
            </div>
        </div>
        <script>
            async function send() {
                const input = document.getElementById('msg');
                const btn = document.getElementById('sendBtn');
                const chat = document.getElementById('chat');
                const text = input.value.trim();
                if (!text) return;

                chat.innerHTML += `<div class="message user">${escapeHtml(text)}</div>`;
                const thinking = document.createElement('div');
                thinking.className = 'thinking';
                thinking.textContent = '⏳ Searching corpus and generating answer…';
                chat.appendChild(thinking);
                input.value = '';
                btn.disabled = true;
                chat.scrollTop = chat.scrollHeight;

                try {
                    const res = await fetch('/query', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({question: text})
                    });
                    const data = await res.json();
                    thinking.remove();

                    let html = `<div class="message ai">${escapeHtml(data.answer)}`;

                    if (data.citations && data.citations.length > 0) {
                        const unique = [...new Set(data.citations.map(c => `${c.source} p.${c.page}`))];
                        html += `<div class="citations"><b>Sources:</b> ${unique.join(' · ')}</div>`;
                    }

                    if (data.latencies) {
                        html += '<div class="latency-row">';
                        const labels = {retrieval_ms:'retrieval', reranking_ms:'rerank', parent_swap_ms:'parent-swap', generation_ms:'llm'};
                        for (const [k, label] of Object.entries(labels)) {
                            if (data.latencies[k] !== undefined) {
                                html += `<span class="latency-pill">${label}: ${Math.round(data.latencies[k])}ms</span>`;
                            }
                        }
                        html += `<span class="latency-pill latency-total">total: ${Math.round(data.total_latency_ms)}ms</span>`;
                        html += '</div>';
                    }
                    html += '</div>';
                    chat.innerHTML += html;
                } catch (e) {
                    thinking.remove();
                    chat.innerHTML += `<div class="message ai" style="color:#f87171">Error: ${escapeHtml(e.message)}</div>`;
                } finally {
                    btn.disabled = false;
                    chat.scrollTop = chat.scrollHeight;
                }
            }

            function escapeHtml(str) {
                return String(str).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
            }
        </script>
    </body>
    </html>
    """


# ─────────────────────────────────────────────────────────────────────────────
# API Endpoints
# ─────────────────────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


@app.post("/query", summary="Run RAG pipeline")
async def query_rag(request: QueryRequest):
    """
    Run the full hybrid RAG pipeline and return the answer, citations, and latency breakdown.
    """
    if not state.pipeline:
        raise HTTPException(status_code=503, detail="Pipeline is still loading — try again in a moment.")
    result = state.pipeline.run(request.question, top_k=request.top_k)
    return {
        "answer": result["answer"],
        "citations": result["citations"],
        "latencies": result["latencies"],
        "total_latency_ms": result["total_latency_ms"],
    }


@app.get("/health", summary="Health check")
async def health():
    """Returns service status and confirms all components loaded correctly."""
    if not state.pipeline or not state.ready:
        return {"status": "loading"}
    p = state.pipeline
    return {
        "status": "ready",
        "startup_ms": round(state.startup_time, 1),
        "faiss_vectors": p.indexer.faiss.ntotal if p.indexer.faiss else 0,
        "bm25_loaded": p.indexer.bm25 is not None,
        "db_connected": p.indexer.db_conn is not None,
        "dense_only": p.dense_only,
        "skip_rerank": p.skip_rerank,
    }


@app.get("/metrics", summary="Evaluation metrics")
async def metrics():
    """Returns RAGAS scores and p99 latency for all three experiment variants."""
    if METRICS_CACHE.exists():
        with open(METRICS_CACHE) as f:
            return json.load(f)
    # Placeholder until run_experiments.py has been executed
    return {
        "note": "Run run_experiments.py to populate real metrics.",
        "variants": [
            {
                "name": "1. Baseline (Dense Only)",
                "ragas_context_precision": None,
                "ragas_faithfulness": None,
                "latency_p99_ms": None,
            },
            {
                "name": "2. Dense + Reranker",
                "ragas_context_precision": None,
                "ragas_faithfulness": None,
                "latency_p99_ms": None,
            },
            {
                "name": "3. Hybrid + Reranker (Full)",
                "ragas_context_precision": None,
                "ragas_faithfulness": None,
                "latency_p99_ms": None,
            },
        ],
    }


@app.get("/experiments", response_class=HTMLResponse, summary="Experiment comparison dashboard")
async def experiments():
    """Visual comparison of the three RAG pipeline variants."""
    data = {"variants": []}
    if METRICS_CACHE.exists():
        with open(METRICS_CACHE) as f:
            data = json.load(f)

    rows = ""
    for v in data.get("variants", []):
        cp = v.get("ragas_context_precision") or v.get("ragas_context_precision")
        fa = v.get("ragas_faithfulness")
        p99 = v.get("latency_p99_ms")
        rows += f"""
        <tr>
            <td>{v.get('name','—')}</td>
            <td>{f"{cp:.3f}" if cp is not None else "—"}</td>
            <td>{f"{fa:.3f}" if fa is not None else "—"}</td>
            <td>{f"{p99:.0f}ms" if p99 is not None else "—"}</td>
        </tr>"""

    return f"""
    <!DOCTYPE html><html lang="en"><head>
    <meta charset="UTF-8"><title>Experiment Comparison</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
        body {{ font-family: 'Inter', sans-serif; background: #0f172a; color: #f8fafc; padding: 2rem; }}
        h1 {{ font-size: 1.5rem; margin-bottom: 0.5rem; }}
        p {{ color: #94a3b8; margin-bottom: 2rem; font-size: 0.9rem; }}
        table {{ width: 100%; border-collapse: collapse; max-width: 800px; }}
        th {{ background: #1e293b; padding: 0.8rem 1rem; text-align: left; font-size: 0.8rem; text-transform: uppercase; color: #94a3b8; letter-spacing: 0.06em; }}
        td {{ padding: 0.9rem 1rem; border-bottom: 1px solid #1e293b; font-size: 0.9rem; }}
        tr:hover td {{ background: #1e293b33; }}
        a {{ color: #818cf8; font-size: 0.85rem; }}
    </style>
    </head><body>
    <h1>🧪 RAG Experiment Comparison</h1>
    <p>Three pipeline variants evaluated on 100 synthetic QA pairs from the ArXiv corpus.
       Run <code>python run_experiments.py</code> to populate real scores.</p>
    <table>
        <thead><tr><th>Variant</th><th>Context Precision ↑</th><th>Faithfulness ↑</th><th>P99 Latency ↓</th></tr></thead>
        <tbody>{rows if rows else "<tr><td colspan='4' style='color:#94a3b8'>No data yet — run run_experiments.py</td></tr>"}</tbody>
    </table>
    <br><a href="/">← Back to Chat</a> | <a href="/docs">Swagger API →</a>
    </body></html>
    """
