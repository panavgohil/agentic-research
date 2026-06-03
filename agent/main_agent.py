from agent.planner import plan
from agent.retriever import retrieve
from agent.reflector import reflect
from agent.synthesizer import synthesize


question = input(
    "Research Question: "
)

print("\nPlanning...\n")

subquestions = plan(question)

for i, q in enumerate(subquestions, start=1):
    print(f"{i}. {q}")

all_evidence = []

print("\nRetrieving Evidence...\n")

for q in subquestions:

    evidence = retrieve(
        q,
        k=3
    )

    all_evidence.extend(
        evidence
    )

reflection = reflect(
    question,
    all_evidence
)

print("\nReflection Result:\n")
print(reflection)

if not reflection.get(
    "enough_evidence",
    True
):

    print(
        "\nGathering More Evidence...\n"
    )

    more_evidence = retrieve(
        reflection["new_query"],
        k=5
    )

    all_evidence.extend(
        more_evidence
    )

answer = synthesize(
    question,
    all_evidence
)

print("\nANSWER\n")
print(answer)