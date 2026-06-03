import arxiv
import csv
import os
import requests
import time

os.makedirs("data/papers", exist_ok=True)

client = arxiv.Client()

queries = [
    "LLM agent",
    "agent memory",
    "agentic RAG",
    "tool use",
    "multi-agent system"
]

csv_file = "data/paper_metadata.csv"

downloaded_ids = set()

with open(csv_file, "w", newline="", encoding="utf-8") as file:

    writer = csv.writer(file)

    writer.writerow([
        "arxiv_id",
        "title",
        "published",
        "abstract",
        "pdf_url"
    ])

    for query in queries:

        print(f"\nSearching: {query}\n")

        search = arxiv.Search(
            query=query,
            max_results=100,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )

        try:

            for result in client.results(search):

                arxiv_id = result.get_short_id()

                if arxiv_id in downloaded_ids:
                    continue

                downloaded_ids.add(arxiv_id)

                print(f"Downloading: {result.title}")

                title = result.title
                published = result.published
                abstract = result.summary
                pdf_url = result.pdf_url

                writer.writerow([
                    arxiv_id,
                    title,
                    published,
                    abstract,
                    pdf_url
                ])

                pdf_path = f"data/papers/{arxiv_id}.pdf"

                if os.path.exists(pdf_path):
                    print("Already exists")
                    continue

                try:

                    response = requests.get(
                        pdf_url,
                        timeout=60
                    )

                    with open(
                        pdf_path,
                        "wb"
                    ) as pdf_file:

                        pdf_file.write(
                            response.content
                        )

                    print(
                        f"Saved: {pdf_path}"
                    )

                    time.sleep(3)

                except Exception as e:

                    print(
                        f"Failed download: {arxiv_id}"
                    )

                    print(e)

        except Exception as e:

            print(
                f"Query failed: {query}"
            )

            print(e)

            time.sleep(30)

print("\nDownload Complete")
print(
    f"Total Unique Papers: {len(downloaded_ids)}"
)