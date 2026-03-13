"""
Classify papers against theses using zero-shot NLI.
Model: cross-encoder/nli-deberta-v3-large

For each (abstract, thesis) pair the model returns:
  entailment   → abstract supports the thesis
  neutral      → abstract does not clearly address the thesis
  contradiction → abstract opposes the thesis

Output: ArtCon_theses.csv
  One column per thesis with the label (entailment/neutral/contradiction)
  + one column per thesis with the entailment score (0–1)
"""
import csv
import re
import torch
from tqdm import tqdm
from transformers import pipeline
from theses import THESES

INPUT   = "ArtCon_keywords.csv"
OUTPUT  = "ArtCon_theses.csv"
MODEL   = "cross-encoder/nli-deberta-v3-large"
BATCH   = 8   # reduce to 4 if OOM


def slugify(text: str) -> str:
    """Turn a thesis into a short column-safe key."""
    text = text.lower().rstrip(".")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text[:60]


THESIS_KEYS = [slugify(t) for t in THESES]


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


def classify_batch(pipe, abstracts: list[str], thesis: str) -> list[dict]:
    """Return list of {label, score} dicts for each abstract."""
    results = pipe(
        abstracts,
        candidate_labels=["entailment", "neutral", "contradiction"],
        hypothesis_template="{}",
        multi_label=False,
    )
    # pipe returns a list when given a list
    if isinstance(results, dict):
        results = [results]
    out = []
    for r in results:
        label_scores = dict(zip(r["labels"], r["scores"]))
        best_label = r["labels"][0]
        entailment_score = label_scores.get("entailment", 0.0)
        out.append({"label": best_label, "score": round(entailment_score, 4)})
    return out


def main():
    import os
    path = INPUT if os.path.exists(INPUT) else "ArtCon.csv"
    print(f"Loading entries from {path}…")
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    abstracts = [row.get("abstract", "") or "" for row in rows]

    pipe = load_pipeline()

    # Results: thesis_key → list of {label, score} per row
    results = {key: [None] * len(rows) for key in THESIS_KEYS}

    for thesis, key in zip(THESES, THESIS_KEYS):
        print(f"\nThesis: {thesis}")
        # Substitute thesis into hypothesis template manually
        for i in tqdm(range(0, len(abstracts), BATCH), unit="batch"):
            batch = abstracts[i:i + BATCH]
            # Replace empty abstracts with a placeholder
            batch_clean = [a if a.strip() else "No abstract available." for a in batch]
            batch_results = pipe(
                batch_clean,
                candidate_labels=[thesis],
                hypothesis_template="This text supports the view that: {}",
                multi_label=False,
            )
            if isinstance(batch_results, dict):
                batch_results = [batch_results]
            for j, r in enumerate(batch_results):
                score = round(r["scores"][0], 4)
                # Also get contradiction: run second pass with negation
                results[key][i + j] = {"score": score}

        # Second pass: get full entailment/neutral/contradiction
        print(f"  Getting full labels…")
        for i in tqdm(range(0, len(abstracts), BATCH), unit="batch"):
            batch = [a if a.strip() else "No abstract available." for a in abstracts[i:i + BATCH]]
            batch_results = pipe(
                batch,
                candidate_labels=["supports", "is neutral about", "contradicts"],
                hypothesis_template=f"This abstract {{}} the thesis: {thesis}",
                multi_label=False,
            )
            if isinstance(batch_results, dict):
                batch_results = [batch_results]
            for j, r in enumerate(batch_results):
                label_map = {"supports": "support", "is neutral about": "neutral", "contradicts": "contradiction"}
                best = r["labels"][0]
                results[key][i + j]["label"] = label_map[best]
                results[key][i + j]["label_score"] = round(r["scores"][0], 4)

    # Write output CSV
    fieldnames = list(rows[0].keys())
    for key in THESIS_KEYS:
        fieldnames += [f"{key}__label", f"{key}__score"]

    with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, row in enumerate(rows):
            for key in THESIS_KEYS:
                r = results[key][i] or {}
                row[f"{key}__label"] = r.get("label", "")
                row[f"{key}__score"] = r.get("label_score", "")
            writer.writerow(row)

    print(f"\nDone → {OUTPUT}")
    print(f"Columns added: {len(THESIS_KEYS) * 2} ({len(THESIS_KEYS)} theses × label + score)")


if __name__ == "__main__":
    main()
