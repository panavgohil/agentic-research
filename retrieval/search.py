import os
import json
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer

# Load model
model = SentenceTransformer(
    "BAAI/bge-small-en-v1.5"
)

# Load FAISS
index = faiss.read_index(
    "data/faiss_index.bin"
)

# Load metadata
with open(
    "data/chunk_metadata.json",
    "r",
    encoding="utf-8"
) as f:
    metadata = json.load(f)

query = input("Ask a question: ")

query_embedding = model.encode(
    [query]
).astype("float32")

k = 5

distances, indices = index.search(
    query_embedding,
    k
)

print("\n" + "=" * 80)
print("TOP RESULTS")
print("=" * 80)

for rank, idx in enumerate(indices[0]):

    item = metadata[idx]

    paper_file = item["paper"]

    chunk_file = os.path.join(
        "data/chunks",
        paper_file
    )

    with open(
        chunk_file,
        "r",
        encoding="utf-8"
    ) as f:

        chunks = json.load(f)

    chunk_text = chunks[
        item["chunk_id"]
    ]["text"]

    print(f"\nResult {rank+1}")
    print(f"Paper : {paper_file}")
    print(f"Chunk : {item['chunk_id']}")

    print("-" * 80)

    print(
        chunk_text[:1000]
    )

    print("-" * 80)