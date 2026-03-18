"""
Compute bibliographic coupling matrix from fetched OpenAlex references.

Bibliographic coupling score(i, j) = number of references shared by papers i and j.

Output:
    ArtCon_coupling.npz   — sparse coupling matrix (696×696, scipy CSR)
    ArtCon_coupling.csv   — non-zero pairs: idx_i, idx_j, score  (for inspection)

Usage:
    uv run python3 compute_coupling.py
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from collections import defaultdict

REFERENCES_FILE = "ArtCon_references.csv"
CLUSTERS_FILE   = "ArtCon_clusters.csv"
OUTPUT_NPZ      = "ArtCon_coupling.npz"
OUTPUT_CSV      = "ArtCon_coupling.csv"


def main():
    # Load paper index (row position → openalex_id)
    papers = pd.read_csv(CLUSTERS_FILE).fillna("")
    n = len(papers)
    id_to_idx = {oid: i for i, oid in enumerate(papers["openalex_id"])
                 if oid.startswith("https://openalex.org/")}
    print(f"Papers: {n}, with OpenAlex ID: {len(id_to_idx)}")

    # Load references
    refs_df = pd.read_csv(REFERENCES_FILE).fillna("")
    print(f"Reference rows loaded: {len(refs_df)}")

    # Build: cited_work → list of paper indices that cite it (inverted index)
    inverted: dict[str, list[int]] = defaultdict(list)
    total_refs = 0
    for _, row in refs_df.iterrows():
        oid = row["openalex_id"]
        if oid not in id_to_idx:
            continue
        idx = id_to_idx[oid]
        refs = [r.strip() for r in str(row["references"]).split("|") if r.strip()]
        for ref in refs:
            inverted[ref].append(idx)
        total_refs += len(refs)

    print(f"Unique cited works: {len(inverted)}, total reference links: {total_refs}")

    # Compute coupling scores via inverted index
    # For each cited work shared by k papers, add 1 to all k*(k-1)/2 pairs
    coupling: dict[tuple[int,int], int] = defaultdict(int)
    for citers in inverted.values():
        if len(citers) < 2:
            continue
        citers_sorted = sorted(set(citers))
        for a in range(len(citers_sorted)):
            for b in range(a + 1, len(citers_sorted)):
                coupling[(citers_sorted[a], citers_sorted[b])] += 1

    print(f"Non-zero pairs: {len(coupling)}")

    # Statistics
    scores = np.array(list(coupling.values()))
    print(f"Coupling scores — min: {scores.min()}, max: {scores.max()}, "
          f"mean: {scores.mean():.2f}, median: {np.median(scores):.1f}")
    for thr in [1, 2, 3, 5, 10, 20]:
        print(f"  pairs with score >= {thr:2d}: {(scores >= thr).sum()}")

    # Build sparse matrix
    rows, cols, data = [], [], []
    for (i, j), s in coupling.items():
        rows += [i, j]
        cols += [j, i]
        data += [s, s]
    mat = sp.csr_matrix((data, (rows, cols)), shape=(n, n))

    sp.save_npz(OUTPUT_NPZ, mat)
    print(f"\nWritten {OUTPUT_NPZ}  (shape {mat.shape}, {len(coupling)} unique pairs)")

    # Save readable CSV (upper triangle only, sorted by score)
    pairs = sorted(coupling.items(), key=lambda x: -x[1])
    pair_rows = [{"idx_i": i, "idx_j": j, "score": s,
                  "title_i": papers.iloc[i]["title"],
                  "title_j": papers.iloc[j]["title"]}
                 for (i, j), s in pairs]
    pd.DataFrame(pair_rows).to_csv(OUTPUT_CSV, index=False)
    print(f"Written {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
