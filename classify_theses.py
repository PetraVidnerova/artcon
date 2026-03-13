"""
Classify papers against thesis pairs using zero-shot NLI.
Model: cross-encoder/nli-deberta-v3-large

For each (abstract, thesis pair) the model picks one of:
  supports   → abstract argues for the topic
  opposes    → abstract argues against the topic
  neutral    → abstract does not clearly take a stance

This ensures mutual exclusivity: a paper cannot support both
a thesis and its negation.

Output columns per pair:  <key>__stance  (supports/opposes/neutral)
                           <key>__score   (confidence 0–1)
"""
import csv
import os
import torch
from tqdm import tqdm
from transformers import pipeline
from theses import THESIS_PAIRS

INPUT   = "ArtCon_keywords.csv"
OUTPUT  = "ArtCon_theses.csv"
MODEL   = "cross-encoder/nli-deberta-v3-large"
BATCH   = 8


def load_pipeline():
    print(f"Loading {MODEL}…")
    device = 0 if torch.cuda.is_available() else -1
    pipe = pipeline(
        "zero-shot-classification",
        model=MODEL,
        device=device,
    )
    print(f"  Running on: {'GPU' if device == 0 else 'CPU'}")
    return pipe


def classify_batch(pipe, abstracts: list[str], pair: dict) -> list[dict]:
    """
    Classify a batch of abstracts against a thesis pair.
    Returns list of {stance, score} dicts.
    """
    topic = pair["topic"]
    pro   = pair["pro"]
    con   = pair["con"]

    results = pipe(
        abstracts,
        candidate_labels=[
            f"supports the view that {pro}",
            f"opposes the view that {pro} and argues that {con}",
            f"does not take a clear stance on {topic}",
        ],
        hypothesis_template="This abstract {}",
        multi_label=False,
    )
    if isinstance(results, dict):
        results = [results]

    out = []
    for r in results:
        best = r["labels"][0]
        score = round(r["scores"][0], 4)
        if "supports" in best:
            stance = "supports"
        elif "opposes" in best:
            stance = "opposes"
        else:
            stance = "neutral"
        out.append({"stance": stance, "score": score})
    return out


def main():
    path = INPUT if os.path.exists(INPUT) else "ArtCon.csv"
    print(f"Loading entries from {path}…")
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    abstracts = [
        (row.get("abstract", "") or "").strip() or "No abstract available."
        for row in rows
    ]

    pipe = load_pipeline()

    # results[pair_key] = list of {stance, score} per row
    results = {pair["key"]: [None] * len(rows) for pair in THESIS_PAIRS}

    for pair in THESIS_PAIRS:
        print(f"\nPair: {pair['topic']}")
        key = pair["key"]
        for i in tqdm(range(0, len(abstracts), BATCH), unit="batch"):
            batch = abstracts[i:i + BATCH]
            batch_results = classify_batch(pipe, batch, pair)
            results[key][i:i + len(batch)] = batch_results

    # Write output
    fieldnames = list(rows[0].keys())
    for pair in THESIS_PAIRS:
        fieldnames += [f"{pair['key']}__stance", f"{pair['key']}__score"]

    with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, row in enumerate(rows):
            for pair in THESIS_PAIRS:
                r = results[pair["key"]][i] or {}
                row[f"{pair['key']}__stance"] = r.get("stance", "")
                row[f"{pair['key']}__score"]  = r.get("score", "")
            writer.writerow(row)

    print(f"\nDone → {OUTPUT}")
    print(f"Columns: {len(THESIS_PAIRS)} pairs × (stance + score)")


if __name__ == "__main__":
    main()
