import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


def synthesize(question, evidence):

    context = ""

    for item in evidence:

        context += f"""
Paper: {item['paper']}
Chunk: {item['chunk_id']}

{item['text']}

------------------
"""

    prompt = f"""
Answer the research question using only
the evidence below.

Research Question:
{question}

Evidence:
{context}

Give a detailed research answer.

At the end provide a concise summary.
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    answer = response.text

    citations = []
    seen = set()

    for item in evidence:

        paper = item["paper"]

        if paper not in seen:

            citations.append(paper)
            seen.add(paper)

    answer += "\n\nREFERENCES\n"

    for i, paper in enumerate(
        citations,
        start=1
    ):

        answer += f"[{i}] {paper}\n"

    return answer


if __name__ == "__main__":

    sample_question = "What is agent memory?"

    sample_evidence = [
        {
            "paper": "sample_paper",
            "chunk_id": 0,
            "text": "Agent memory helps retain information across tasks."
        }
    ]

    print(
        synthesize(
            sample_question,
            sample_evidence
        )
    )