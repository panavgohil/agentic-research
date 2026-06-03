# Agentic Deep Research — AIMS DTU Research Intern 2026

An agentic deep-research system over a corpus of 19 recent LLM-agent arXiv papers, with ablation study of each architectural component.

## Quick Start (Single Command)

```bash
# 1. Install dependencies
pip install google-generativeai sentence-transformers faiss-cpu rank-bm25 python-dotenv

# 2. Set your Gemini API key
echo "GEMINI_API_KEY=your_key_here" > .env

# 3. Run evaluation (all 6 configs × 30 questions → predictions/ + scores)
python run_eval.py
```

To run a single config:
```bash
python run_eval.py --config full_agent
python run_eval.py --config baseline
```

To score existing predictions without re-running inference:
```bash
python run_eval.py --score
```

To ask a single question interactively:
```bash
python -m agent.agent --question "What is the CRANMEM memory architecture?"
python -m agent.agent --question "..." --no-planner     # ablation: no planner
python -m agent.agent --question "..." --no-reflector   # ablation: no reflector
python -m agent.agent --question "..." --baseline       # non-agentic single-shot
```

## Repository Layout

```
agentic-research/
├── agent/
│   ├── agent.py          # Full agent with ablation flags
│   ├── planner.py        # Sub-question decomposition (legacy)
│   ├── reflector.py      # Evidence sufficiency check (legacy)
│   ├── retriever.py      # FAISS dense retrieval (legacy)
│   └── synthesizer.py    # Answer synthesis (legacy)
├── data/
│   ├── papers/           # 19 PDFs (arXiv)
│   ├── metadata/         # Extracted plain text
│   ├── chunks/           # Word-windowed JSON chunks
│   ├── chunk_metadata.json
│   └── faiss_index.bin
├── eval/
│   ├── questions.jsonl   # 30 evaluation questions
│   ├── SUBMISSION_FORMAT.md
│   └── scores.json       # Written by run_eval.py --score
├── indexing/
│   ├── chunker.py        # Text → chunks
│   └── embeddings.py     # Chunks → FAISS index
├── predictions/          # Output: one .jsonl per config
├── retrieval/
│   └── search.py         # Standalone retrieval demo
├── scraper/
│   └── arxiv_scraper.py  # arXiv collection script
├── run_eval.py           # Evaluation runner
└── .env                  # GEMINI_API_KEY=...
```

## Architecture

```
Question
  │
  ▼
[Planner]  ──── decomposes into 3-5 sub-questions
  │
  ▼
[Retriever] ─── hybrid dense (FAISS) + sparse (BM25) RRF
  │              k=5 chunks per sub-question
  ▼
[Reflector] ─── is evidence sufficient? (Gemini JSON output)
  │              if not → one more retrieval pass
  ▼
[Synthesizer] ── grounded answer with [arXiv ID] citations (Gemini)
  │
  ▼
[Citation Verifier] ── drops unsupported citations (vocab overlap)
  │
  ▼
Final Answer + Citations + Trace
```

## Ablation Configurations

| Config | Planner | Reflector | Cit-Verifier | Hybrid |
|--------|---------|-----------|--------------|--------|
| `full_agent` | ✅ | ✅ | ✅ | ✅ |
| `baseline` | ❌ | ❌ | ❌ | ✅ (single-shot) |
| `no_planner` | ❌ | ✅ | ✅ | ✅ |
| `no_reflector` | ✅ | ❌ | ✅ | ✅ |
| `no_citation_verifier` | ✅ | ✅ | ❌ | ✅ |
| `no_hybrid` | ✅ | ✅ | ✅ | ❌ (dense-only) |

## Rebuilding the Index from Scratch

```bash
# Parse PDFs → text
python indexing/pdf_parser.py

# Chunk text
python indexing/chunker.py

# Build FAISS index
python indexing/embeddings.py
```

## Key Design Choices

- **Embedding model**: `all-MiniLM-L6-v2` (384-d, fast, free)
- **Vector store**: FAISS `IndexFlatL2` (exact search, appropriate for 339 vectors)
- **Hybrid retrieval**: BM25Okapi + FAISS via Reciprocal Rank Fusion (k=60, α=0.5)
- **LLM backend**: Gemini 2.5 Flash (free tier)
- **Chunking**: Fixed 500-word windows (reproducible, no boundary heuristics)
- **Citation verification**: Vocabulary overlap heuristic (lightweight, no extra API calls)

## Dependencies

```
google-generativeai>=0.8
sentence-transformers>=3.0
faiss-cpu>=1.8
rank-bm25>=0.2
python-dotenv>=1.0
numpy>=1.26
```
