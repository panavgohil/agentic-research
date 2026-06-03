# Agentic Deep Research System

## Project Overview

This project implements an agent-based research assistant that answers questions using a collection of research papers. The system retrieves relevant information from indexed papers and generates a response supported by citations.

The project was developed as part of the AIMS DTU Research Internship 2026 assignment on Agentic AI and Retrieval-Augmented Generation (RAG).

## Features

* Research paper collection from arXiv
* PDF parsing and text extraction
* Document chunking and indexing
* Semantic search using embeddings
* FAISS-based vector retrieval
* Planning agent for question decomposition
* Reflection agent for evidence checking
* Answer generation using Gemini
* Citation-supported responses
* Evaluation and ablation study support

## Dataset

The current implementation was tested on a collection of 19 research papers related to:

* LLM Agents
* Agent Memory
* Tool Use
* Agentic RAG
* Multi-Agent Systems

The system architecture is designed so that larger paper collections can be indexed and searched using the same pipeline.

## Project Structure

agent/

* planner.py
* retriever.py
* reflector.py
* synthesizer.py
* agent.py

data/

* papers/
* chunks/
* metadata/
* faiss_index.bin

eval/

* questions.jsonl
* scores.json

predictions/

* generated outputs

indexing/

* pdf_parser.py
* chunker.py
* embeddings.py

scraper/

* arxiv_scraper.py

## System Workflow

1. User enters a research question.
2. The planner breaks the question into smaller sub-questions.
3. Relevant chunks are retrieved from the indexed paper collection.
4. The reflector checks whether enough evidence has been collected.
5. Additional retrieval is performed if required.
6. The synthesizer generates a final answer with citations.

## Technologies Used

* Python
* FAISS
* Sentence Transformers
* Gemini API
* NumPy
* Pandas

## Running the Project

Install dependencies:

pip install -r requirements.txt

Run the main system:

python -m agent.main_agent

Run evaluation:

python run_eval.py

## Evaluation

The project includes evaluation questions and supports different ablation configurations to compare the contribution of individual components such as:

* Planner
* Reflector
* Citation Verifier
* Hybrid Retrieval

## Author

Panav Gohil
