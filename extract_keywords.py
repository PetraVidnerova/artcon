"""
Extract keyphrases from abstracts using:
  ml6team/keyphrase-extraction-kbir-inspec
  (token classification, trained on scientific abstracts)

Output: ArtCon_keywords.csv — original CSV + 'keywords' column (semicolon-separated)
"""
import csv
import torch
from tqdm import tqdm
from transformers import pipeline

INPUT   = "ArtCon_openalex.csv"
OUTPUT  = "ArtCon_keywords.csv"
MODEL   = "ml6team/keyphrase-extraction-kbir-inspec"
BATCH   = 16


def load_pipeline():
    print(f"Loading model {MODEL}…")
    device = 0 if torch.cuda.is_available() else -1
    pipe = pipeline(
        "token-classification",
        model=MODEL,
        aggregation_strategy="simple",
        device=device,
    )
    print(f"  Running on: {'GPU' if device == 0 else 'CPU'}")
    return pipe


def extract_keywords(pipe, text: str) -> list[str]:
    if not text.strip():
        return []
    results = pipe(text)
    # Deduplicate, lowercase, filter low-confidence spans
    seen = set()
    keywords = []
    for r in results:
        kw = r["word"].strip().lower()
        if kw and r["score"] >= 0.5 and kw not in seen:
            seen.add(kw)
            keywords.append(kw)
    return keywords


def main():
    import os
    path = INPUT if os.path.exists(INPUT) else "ArtCon.csv"
    print(f"Loading entries from {path}…")
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    pipe = load_pipeline()

    # Process in batches for speed
    abstracts = [row.get("abstract", "") or "" for row in rows]
    all_keywords = [""] * len(rows)

    print(f"Extracting keywords from {len(rows)} abstracts…")
    for i in tqdm(range(0, len(abstracts), BATCH), unit="batch"):
        batch = abstracts[i:i + BATCH]
        # Skip empty abstracts
        results = []
        for text in batch:
            kws = extract_keywords(pipe, text)
            results.append("; ".join(kws))
        all_keywords[i:i + len(batch)] = results

    # Write output
    fieldnames = list(rows[0].keys()) + ["keywords"]
    with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row, kws in zip(rows, all_keywords):
            row["keywords"] = kws
            writer.writerow(row)

    n_with_kws = sum(1 for k in all_keywords if k)
    print(f"\nDone. Keywords extracted for {n_with_kws}/{len(rows)} entries.")
    print(f"Output: {OUTPUT}")


if __name__ == "__main__":
    main()
