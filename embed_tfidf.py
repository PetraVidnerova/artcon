"""
Generate TF-IDF embeddings for each entry in ArtCon.csv (or ArtCon_openalex.csv).
Input:  title + abstract per row
Output: embeddings saved as .npy (shape: N x V) + index CSV mapping row → entry

This is a lexical counterpart to embed_specter2.py: instead of contextual
SPECTER2 vectors it produces sparse-bag-of-words TF-IDF vectors, densified and
L2-normalised so cosine similarity = dot product (matching the convention used
by the SPECTER2 embedding graphs). Row order matches specter2_index.csv, so the
output drops straight into the clustering / comparison / viewer machinery as a
third graph type alongside the original and fine-tuned SPECTER2 graphs.

Usage:
    uv run python3 embed_tfidf.py
    uv run python3 embed_tfidf.py --max-features 20000 --ngram-max 2
"""
import argparse
import csv
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

INPUT      = "ArtCon_openalex.csv"   # falls back to ArtCon.csv if not found
OUTPUT_NPY = "tfidf_embeddings.npy"
OUTPUT_IDX = "tfidf_index.csv"


def load_rows():
    path = INPUT if os.path.exists(INPUT) else "ArtCon.csv"
    print(f"Loading entries from {path}…")
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"  {len(rows)} entries loaded")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-features", type=int, default=20000,
                    help="Cap on vocabulary size (most frequent terms kept). "
                         "Keeps the dense .npy a reasonable size (default 20000)")
    ap.add_argument("--ngram-max", type=int, default=2,
                    help="Max n-gram length; 1=unigrams, 2=uni+bigrams (default 2)")
    ap.add_argument("--min-df", type=int, default=2,
                    help="Ignore terms appearing in fewer than this many docs (default 2)")
    args = ap.parse_args()

    rows = load_rows()

    # Same text SPECTER2 sees: title + abstract (here joined with a space, since
    # TF-IDF is order-insensitive and has no SEP token).
    texts = [
        ((row.get("title") or "") + " " + (row.get("abstract") or "")).strip()
        for row in rows
    ]
    n_empty = sum(1 for t in texts if not t)
    print(f"  {n_empty} entries with empty title+abstract")

    # Write index CSV (identical schema to specter2_index.csv for interchangeability)
    index_fields = ["row_index", "authors", "year", "title", "doi", "openalex_id"]
    with open(OUTPUT_IDX, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=index_fields, extrasaction="ignore")
        writer.writeheader()
        for i, row in enumerate(rows):
            writer.writerow({"row_index": i, **row})

    print(f"Fitting TF-IDF (max_features={args.max_features}, "
          f"ngram_range=(1,{args.ngram_max}), min_df={args.min_df})…")
    vec = TfidfVectorizer(
        stop_words="english",
        lowercase=True,
        ngram_range=(1, args.ngram_max),
        min_df=args.min_df,
        max_features=args.max_features,
        sublinear_tf=True,   # 1 + log(tf), dampens long abstracts
        norm="l2",           # L2-normalised → cosine similarity == dot product
    )
    mat = vec.fit_transform(texts).toarray().astype(np.float32)
    np.save(OUTPUT_NPY, mat)

    print(f"\nDone. Embeddings shape: {mat.shape}")
    print(f"  {OUTPUT_NPY}  — float32 array, one L2-normalised TF-IDF row per entry")
    print(f"  {OUTPUT_IDX} — index mapping row number to entry metadata")


if __name__ == "__main__":
    main()
