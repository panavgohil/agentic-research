import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

index = faiss.read_index(
    "data/faiss_index.bin"
)

with open(
    "data/chunk_metadata.json",
    "r",
    encoding="utf-8"
) as f:

    metadata = json.load(f)


def retrieve(query, k=5):

    query_embedding = model.encode(
        [query]
    )

    query_embedding = np.array(
        query_embedding,
        dtype=np.float32
    )

    distances, indices = index.search(
        query_embedding,
        k
    )

    results = []

    for idx in indices[0]:

        item = metadata[idx]

        filename = item["paper"]

        if not filename.endswith(".json"):
            filename += ".json"

        with open(
            f"data/chunks/{filename}",
            "r",
            encoding="utf-8"
        ) as f:

            chunks = json.load(f)

        chunk_text = chunks[
            item["chunk_id"]
        ]["text"]

        results.append(
            {
                "paper": item["paper"],
                "chunk_id": item["chunk_id"],
                "text": chunk_text
            }
        )

    return results


if __name__ == "__main__":

    query = input(
        "Ask a question: "
    )

    results = retrieve(
        query,
        k=5
    )

    print("\nTop Results:\n")

    for i, result in enumerate(
        results,
        start=1
    ):

        print(
            f"{i}. {result['paper']} | Chunk {result['chunk_id']}"
        )

        print(
            result["text"][:300]
        )

        print("-" * 80)