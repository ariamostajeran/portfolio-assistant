# Portfolio Assistant

An agentic RAG system that answers questions about Aria Mostajeran's portfolio,
projects, skills, and experience. Visitors ask questions in plain English — the
agent retrieves from a vector store grounded in real CV and project content, then
generates accurate answers via an LLM.

Live at: [aria-portfolio.onrender.com](https://aria-portfolio.onrender.com) *(coming soon)*

---



## Environment Setup

1. **Copy the environment template:**
   ```bash
   cp .env.example .env
   ```

2. **Add your OpenAI API key to `.env`:**
   ```
   OPENAI_API_KEY=your-actual-key-here
   ```

3. **The `.env` file is in `.gitignore`** and will never be committed to git.


## What It Does

- Answers recruiter and employer questions about Aria's background
- Retrieves from two knowledge sources: **project docs + CV** and **actual source code**
- Maintains conversation history across turns (multi-turn memory)
- Refuses to answer when information isn't in the knowledge base (no hallucination)

Example questions it handles:
> *"Does Aria have experience with Docker?"*
> *"What is Aria's strongest ML project?"*
> *"Show me how the data fetcher is implemented"*
> *"Which of Aria's projects used time series data?"*

---

## Architecture

```
knowledge/               ← markdown files (CV + project write-ups)
  cv.md
  projects/
    portfolio_site.md
    stock_mlops.md

Two Chroma collections:
  knowledge_base         ← chunked markdown, embedded with bge-small-en-v1.5
  code_base              ← source code chunked by function/class (AST + regex)

Agent (ReAct loop):
  search_knowledge       ← semantic search over knowledge_base
  get_cv_section         ← exact CV section lookup
  search_code            ← semantic search over code_base

API:
  POST /api/chat         ← send message, get answer
  POST /api/chat/reset   ← new conversation
  GET  /api/chat/status  ← health check
  GET  /projects/assistant ← chat UI page
```

---

## Tech Stack

| Component | Choice | Why |
|---|---|---|
| Embeddings | `bge-small-en-v1.5` | Free, open source, 384-dim, fast |
| Vector store | ChromaDB 0.5.23 | Embedded, no extra infra needed |
| LLM | GPT-3.5-turbo | Simple, cheap (plan: swap to Mistral 7B) |
| Agent pattern | ReAct (text-based) | Simple, debuggable, model-agnostic |
| Code chunking | AST (Python), regex (JS) | Respects function boundaries |
| Backend | Flask | Integrates into existing portfolio |
| Dev infra | Databricks | Compute + storage during development |

---

## Project Structure

```
portfolio-assistant/
├── knowledge/
│   ├── cv.md
│   └── projects/
│       ├── portfolio_site.md
│       └── stock_mlops.md
├── ingestion/
│   ├── __init__.py
│   ├── persistence.py
│   ├── ingest_knowledge.py
│   └── ingest_code.py
├── retrieval/
│   ├── __init__.py
│   └── retriever.py
├── agent/
│   ├── __init__.py
│   └── assistant.py
├── api/
│   ├── __init__.py
│   └── routes.py
├── notebooks/
│   ├── 01_ingest_knowledge.ipynb
│   ├── 02_ingest_code.ipynb
│   └── 03_agent.ipynb
├── app.py
├── config.py
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/ariamostajeran/portfolio-assistant.git
cd portfolio-assistant
pip install -r requirements.txt
```

### 2. Environment variables

```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

### 3. Ingest knowledge base

```bash
# Ingest CV + project docs
python -m ingestion.ingest_knowledge

# Ingest source code from portfolio repo
python -m ingestion.ingest_code --repo https://github.com/ariamostajeran/aria-portfolio
```

### 4. Run

```bash
python app.py
# Visit http://localhost:5001/api/chat/status to confirm ingestion worked
```

---

## Adding a New Project

1. Write a markdown file using this template:

```markdown
# Project Name

## Summary
## Tech stack
## What it does
## Key technical decisions
## What I learned
## Results / metrics
## Links
```

2. Drop it in `knowledge/projects/`

3. Run:
```bash
python -m ingestion.ingest_knowledge --file knowledge/projects/new_project.md
```

Done. The assistant knows about the new project immediately.

---

## Databricks Notes

Chroma requires SQLite file locking so it runs in `/tmp` (not `/dbfs/`).
`persistence.py` automatically zips the store to the Workspace after every
ingest and restores it on startup — no manual re-ingestion needed after
cluster restarts.

Known dependency constraints (do not change):
```
chromadb==0.5.23   # 0.6+ Rust backend breaks on Databricks Python 3.12
numpy<2.0          # chromadb 0.5.x uses np.float_ removed in NumPy 2.0
```

---

## Roadmap

- [ ] Chat UI frontend (`assistant.html`)
- [ ] Integrate into main portfolio site (aria-portfolio)
- [ ] Swap GPT-3.5-turbo for Mistral 7B on Databricks (fully open source)
- [ ] Add hybrid search (BM25 + vector) for better recall
- [ ] Streaming responses (SSE) for better UX
- [ ] Dockerize everything
- [ ] Evaluation framework (test Q&A pairs, measure retrieval accuracy)