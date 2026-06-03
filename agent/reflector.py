import os
import json
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


def reflect(question, evidence):

    context = ""

    for item in evidence:

        context += f"""
Paper: {item['paper']}
Chunk: {item['chunk_id']}

{item['text']}

------------------
"""

    prompt = f"""
You are a research reflection agent.

Question:
{question}

Evidence:
{context}

Determine whether the evidence is sufficient.

Return ONLY valid JSON.

Example:

{{
  "enough_evidence": true
}}

or

{{
  "enough_evidence": false,
  "new_query": "better search query"
}}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    try:

        return json.loads(
            response.text.strip()
        )

    except Exception:

        return {
            "enough_evidence": True
        }


if __name__ == "__main__":

    sample = [
        {
            "paper": "test",
            "chunk_id": 0,
            "text": "Agent memory stores information."
        }
    ]

    print(
        reflect(
            "What is agent memory?",
            sample
        )
    )