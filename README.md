# ArXiv Production RAG

A production-structured RAG pipeline for PDF understanding with hybrid retrieval, cross-encoder reranking, and automated evaluation.

## 🚀 Architecture
- **Retriever**: Hybrid (FAISS Dense + BM25 Sparse) with RRF Fusion.
- **Reranker**: Flashrank (ms-marco-MiniLM-L-12-v2 cross-encoder).
- **LLM**: Groq (llama-3.1-8b-instant).
- **Evaluation**: RAGAS + MLflow (via DagsHub).
- **Deployment**: Hugging Face Spaces (Docker).

## 🛠️ Local Setup
1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Configure Environment**:
   Edit `.env` and add your `GROQ_API_KEY`.
3. **Run Local Test**:
   ```bash
   python build_indexes.py --papers 5 --skip-expansion
   ```
4. **Start API**:
   ```bash
   uvicorn app.main:app --reload
   ```

## 🧪 Experiments
Run the full evaluation suite:
```bash
python run_experiments.py
```
View results at `/experiments` on the running API or on your DagsHub MLflow dashboard.

## 🚢 Deployment
Every push to `main` triggers:
1. **Unit Tests**: Logic validation.
2. **Smoke Test**: API startup check.
3. **Auto-deploy**: Pushes to Hugging Face Spaces.
