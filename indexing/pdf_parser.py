import fitz
import os

PDF_FOLDER = "data/papers"
OUTPUT_FOLDER = "data/metadata"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

for filename in os.listdir(PDF_FOLDER):

    if filename.endswith(".pdf"):

        pdf_path = os.path.join(PDF_FOLDER, filename)

        print(f"Reading {filename}")

        try:
            doc = fitz.open(pdf_path)

            text = ""

            for page in doc:
                text += page.get_text()

            txt_filename = filename.replace(".pdf", ".txt")

            with open(
                os.path.join(OUTPUT_FOLDER, txt_filename),
                "w",
                encoding="utf-8"
            ) as f:
                f.write(text)

            print(f"Saved {txt_filename}")

        except Exception as e:
            print(f"Error in {filename}")
            print(e)

print("Parsing Complete!")