import os
import json
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer

CHUNK_FOLDER = "data/chunks"

# Load embedding model
model = SentenceTransformer(
    "BAAI/bge-small-en-v1.5"
)

all_texts = []
metadata = []

# Read chunk files
for filename in os.listdir(CHUNK_FOLDER):

    if filename.endswith(".json"):

        filepath = os.path.join(
            CHUNK_FOLDER,
            filename
        )

        with open(
            filepath,
            "r",
            encoding="utf-8"
        ) as f:

            chunks = json.load(f)

        for chunk in chunks:

            all_texts.append(
                chunk["text"]
            )

            metadata.append({
                "paper": filename,
                "chunk_id": chunk["chunk_id"]
            })

print(f"Total chunks: {len(all_texts)}")

# Generate embeddings
embeddings = model.encode(
    all_texts,
    show_progress_bar=True
)

embeddings = np.array(
    embeddings
).astype("float32")

# Create FAISS index
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(
    dimension
)

index.add(embeddings)

# Save index
faiss.write_index(
    index,
    "data/faiss_index.bin"
)

# Save metadata
with open(
    "data/chunk_metadata.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metadata,
        f,
        indent=2
    )

print("FAISS index saved!")