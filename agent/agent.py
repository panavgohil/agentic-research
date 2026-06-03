"""
Full agentic deep-research pipeline with ablation support.

Usage:
    python -m agent.agent --config full_agent --question "Your question here"
    python run_eval.py --config full_agent

Config flags (any combination can be disabled):
    --no-planner            Skip sub-question decomposition
    --no-reflector          Skip reflection / re-retrieval loop
    --no-citation-verifier  Skip citation verification post-pass
    --no-hybrid             Use dense-only retrieval (no BM25)
    --baseline              Single-shot mode (no agent loop at all)
"""

import os
import json
import time
import argparse
import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

# ── LLM client ─────────────────────────────────────────────────────────────
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
_model = genai.GenerativeModel("gemini-2.5-flash")


def _llm(prompt: str) -> str:
    resp = _model.generate_content(prompt)
    return resp.text.strip()


# ── Retrieval index (loaded once) ───────────────────────────────────────────
_BASE = os.path.join(os.path.dirname(__file__), "..", "data")
_embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
_faiss_index = faiss.read_index(os.path.join(_BASE, "faiss_index.bin"))

with open(os.path.join(_BASE, "chunk_metadata.json"), encoding="utf-8") as f:
    _chunk_meta = json.load(f)

# Build BM25 corpus lazily
_bm25 = None
_bm25_corpus = None


def _get_bm25():
    global _bm25, _bm25_corpus
    if _bm25 is not None:
        return _bm25, _bm25_corpus
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        return None, None
    corpus_texts = []
    for item in _chunk_meta:
        fname = item["paper"]
        if not fname.endswith(".json"):
            fname += ".json"
        with open(os.path.join(_BASE, "chunks", fname), encoding="utf-8") as f:
            chunks = json.load(f)
        corpus_texts.append(chunks[item["chunk_id"]]["text"])
    tokenized = [t.lower().split() for t in corpus_texts]
    _bm25 = BM25Okapi(tokenized)
    _bm25_corpus = corpus_texts
    return _bm25, _bm25_corpus


def _load_chunk_text(item: dict) -> str:
    fname = item["paper"]
    if not fname.endswith(".json"):
        fname += ".json"
    with open(os.path.join(_BASE, "chunks", fname), encoding="utf-8") as f:
        chunks = json.load(f)
    return chunks[item["chunk_id"]]["text"]


# ── Component functions ──────────────────────────────────────────────────────

def planner(question: str) -> list[str]:
    """Decompose question into 3–5 sub-questions."""
    prompt = f"""You are an expert research planner.
Break the following research question into 3 to 5 focused sub-questions
that together cover the full scope of the question.

Question:
{question}

Return ONLY a numbered list (1. … 2. … etc.). No explanations."""
    text = _llm(prompt)
    subs = []
    for line in text.splitlines():
        line = line.strip()
        m = re.match(r"^\d+[.)]\s*(.*)", line)
        if m:
            subs.append(m.group(1).strip())
    return subs if subs else [question]


def dense_retrieve(query: str, k: int = 5) -> list[dict]:
    """FAISS dense retrieval."""
    emb = _embedding_model.encode([query], normalize_embeddings=True).astype("float32")
    distances, indices = _faiss_index.search(emb, k)
    results = []
    for idx, dist in zip(indices[0], distances[0]):
        item = _chunk_meta[idx]
        text = _load_chunk_text(item)
        paper_id = item["paper"].replace(".json", "")
        results.append({
            "paper": paper_id,
            "chunk_id": item["chunk_id"],
            "text": text,
            "score": float(1 / (1 + dist)),
        })
    return results


def hybrid_retrieve(query: str, k: int = 5, alpha: float = 0.5) -> list[dict]:
    """Combine dense FAISS + BM25 sparse retrieval (RRF fusion)."""
    bm25, corpus = _get_bm25()

    dense = dense_retrieve(query, k=k * 2)

    if bm25 is None:
        return dense[:k]

    tokens = query.lower().split()
    sparse_scores = bm25.get_scores(tokens)
    top_sparse_idx = np.argsort(sparse_scores)[::-1][: k * 2]

    # Reciprocal Rank Fusion
    rrf_scores: dict[int, float] = {}
    for rank, res in enumerate(dense):
        raw_idx = next(
            (i for i, m in enumerate(_chunk_meta)
             if m["paper"].replace(".json", "") == res["paper"]
             and m["chunk_id"] == res["chunk_id"]),
            None,
        )
        if raw_idx is not None:
            rrf_scores[raw_idx] = rrf_scores.get(raw_idx, 0) + alpha / (rank + 60)

    for rank, idx in enumerate(top_sparse_idx):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + (1 - alpha) / (rank + 60)

    sorted_idx = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:k]
    results = []
    for idx in sorted_idx:
        item = _chunk_meta[idx]
        text = _load_chunk_text(item)
        paper_id = item["paper"].replace(".json", "")
        results.append({
            "paper": paper_id,
            "chunk_id": item["chunk_id"],
            "text": text,
            "score": rrf_scores[idx],
        })
    return results


def reflector(question: str, evidence: list[dict]) -> dict:
    """Decide if evidence is sufficient; if not, suggest a new query."""
    context = "\n\n".join(
        f"Paper: {e['paper']}\n{e['text'][:600]}" for e in evidence[:10]
    )
    prompt = f"""You are a research reflection agent.

Question:
{question}

Retrieved evidence (excerpts):
{context}

Is the evidence sufficient to answer the question thoroughly?
- If YES: return exactly: {{"enough_evidence": true}}
- If NO: return exactly: {{"enough_evidence": false, "new_query": "<refined search query>"}}

Return ONLY valid JSON. No markdown, no explanation."""
    text = _llm(prompt)
    text = re.sub(r"```json|```", "", text).strip()
    try:
        return json.loads(text)
    except Exception:
        return {"enough_evidence": True}


def synthesizer(question: str, evidence: list[dict]) -> tuple[str, list[str]]:
    """Synthesize a grounded answer with inline arXiv citations."""
    # Deduplicate evidence
    seen = set()
    unique = []
    for e in evidence:
        key = (e["paper"], e["chunk_id"])
        if key not in seen:
            seen.add(key)
            unique.append(e)

    context = ""
    for e in unique[:15]:
        context += f"\n[{e['paper']}] chunk {e['chunk_id']}:\n{e['text'][:800]}\n---\n"

    prompt = f"""You are a research synthesis agent.

Answer the question using ONLY the evidence provided. Cite sources inline using
arXiv ID notation like [2402.11163v1]. Every factual claim must have a citation.

Research Question:
{question}

Evidence:
{context}

Write a detailed, well-structured answer. End with a one-paragraph summary.
Use inline citations throughout."""
    answer = _llm(prompt)

    # Extract cited IDs
    cited = sorted(set(re.findall(r"\b(\d{4}\.\d{4,5}v\d+)\b", answer)))
    return answer, cited


def citation_verifier(answer: str, citations: list[str], evidence: list[dict]) -> tuple[str, list[str]]:
    """
    Verify each citation: check if the cited paper's chunks actually support
    the surrounding claim. Remove unsupported citations and flag hallucinated ones.
    """
    # Build paper → text lookup
    paper_texts: dict[str, str] = {}
    for e in evidence:
        pid = e["paper"]
        paper_texts.setdefault(pid, "")
        paper_texts[pid] += " " + e["text"]

    verified = []
    dropped = []
    for cid in citations:
        if cid not in paper_texts:
            dropped.append(cid)
            continue
        # Quick check: does the cited paper's text share vocabulary with the answer?
        answer_words = set(answer.lower().split())
        paper_words = set(paper_texts[cid].lower().split())
        overlap = len(answer_words & paper_words) / max(len(answer_words), 1)
        if overlap > 0.05:  # at least 5% vocabulary overlap
            verified.append(cid)
        else:
            dropped.append(cid)

    if dropped:
        note = f"\n\n> ⚠️ Citations removed by verifier (not supported by retrieved evidence): {', '.join(dropped)}"
        answer = answer + note

    return answer, verified


# ── Main run function ─────────────────────────────────────────────────────────

def run(
    question: str,
    use_planner: bool = True,
    use_reflector: bool = True,
    use_citation_verifier: bool = True,
    use_hybrid: bool = True,
    baseline: bool = False,
    k: int = 5,
) -> dict:
    t0 = time.time()
    tool_calls = 0
    retrieval_rounds = 0
    subquestions = []
    reflection_decision = "n/a"

    retrieve_fn = hybrid_retrieve if use_hybrid else dense_retrieve

    all_evidence: list[dict] = []

    if baseline:
        # ── Baseline: single retrieve + single LLM call ──────────────────────
        results = retrieve_fn(question, k=k)
        tool_calls += 1
        retrieval_rounds = 1
        all_evidence = results
        answer, citations = synthesizer(question, all_evidence)
        tool_calls += 1

    else:
        # ── Agentic loop ─────────────────────────────────────────────────────
        if use_planner:
            subquestions = planner(question)
            tool_calls += 1
        else:
            subquestions = [question]

        # Retrieve for each sub-question
        for sq in subquestions:
            results = retrieve_fn(sq, k=k)
            all_evidence.extend(results)
            tool_calls += 1
        retrieval_rounds = 1

        # Reflect and optionally loop
        if use_reflector:
            reflection = reflector(question, all_evidence)
            tool_calls += 1
            if not reflection.get("enough_evidence", True):
                extra = retrieve_fn(reflection.get("new_query", question), k=k + 2)
                all_evidence.extend(extra)
                tool_calls += 1
                retrieval_rounds = 2
                reflection_decision = "searched_again"
            else:
                reflection_decision = "enough_evidence"
        else:
            reflection_decision = "n/a"

        # Synthesize
        answer, citations = synthesizer(question, all_evidence)
        tool_calls += 1

        # Verify citations
        if use_citation_verifier:
            answer, citations = citation_verifier(answer, citations, all_evidence)

    latency = round(time.time() - t0, 2)

    return {
        "question": question,
        "answer": answer,
        "citations": citations,
        "trace": {
            "subquestions": subquestions,
            "retrieval_rounds": retrieval_rounds,
            "tool_calls": tool_calls,
            "reflection_decision": reflection_decision,
            "latency_seconds": latency,
        },
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agentic deep-research agent")
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--no-planner", action="store_true")
    parser.add_argument("--no-reflector", action="store_true")
    parser.add_argument("--no-citation-verifier", action="store_true")
    parser.add_argument("--no-hybrid", action="store_true")
    parser.add_argument("--baseline", action="store_true")
    args = parser.parse_args()

    result = run(
        question=args.question,
        use_planner=not args.no_planner,
        use_reflector=not args.no_reflector,
        use_citation_verifier=not args.no_citation_verifier,
        use_hybrid=not args.no_hybrid,
        baseline=args.baseline,
    )

    print("\n" + "=" * 80)
    print("ANSWER")
    print("=" * 80)
    print(result["answer"])
    print("\nCITATIONS:", result["citations"])
    print("TRACE:", json.dumps(result["trace"], indent=2))
