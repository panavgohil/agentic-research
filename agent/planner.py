import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


def plan(question):

    prompt = f"""
You are an expert research planner.

Break the following research question into
3 to 5 smaller research questions.

Question:
{question}

Return only a numbered list.
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    text = response.text

    subquestions = []

    for line in text.split("\n"):

        line = line.strip()

        if not line:
            continue

        if (
            line.startswith("1")
            or line.startswith("2")
            or line.startswith("3")
            or line.startswith("4")
            or line.startswith("5")
        ):

            parts = line.split(".", 1)

            if len(parts) > 1:
                subquestions.append(parts[1].strip())

    return subquestions


if __name__ == "__main__":

    question = input(
        "Research Question: "
    )

    result = plan(question)

    print("\nGenerated Subquestions:\n")

    for i, q in enumerate(result, start=1):
        print(f"{i}. {q}")