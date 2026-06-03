import os
import json

INPUT_FOLDER = "data/metadata"
OUTPUT_FOLDER = "data/chunks"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

CHUNK_SIZE = 500

for filename in os.listdir(INPUT_FOLDER):

    if filename.endswith(".txt"):

        filepath = os.path.join(INPUT_FOLDER, filename)

        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        words = text.split()

        chunks = []

        for i in range(0, len(words), CHUNK_SIZE):

            chunk_text = " ".join(words[i:i + CHUNK_SIZE])

            chunks.append({
                "chunk_id": len(chunks),
                "text": chunk_text
            })

        output_file = filename.replace(".txt", ".json")

        with open(
            os.path.join(OUTPUT_FOLDER, output_file),
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(chunks, f, indent=2)

        print(f"Created {len(chunks)} chunks for {filename}")

print("Chunking Complete!")