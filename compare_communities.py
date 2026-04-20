"""
Compare community structure of the original vs. fine-tuned SPECTER2
similarity graph.

Self-contained: runs Louvain on both thresholded graphs, then reports
  - per-graph: #communities, singleton count, size of largest, modularity
  - cross-graph: Normalized Mutual Information, Adjusted Rand Index
  - embedding-quality (each embedding space using its own Louvain labels,
    singletons dropped): silhouette (cosine), Davies-Bouldin

Usage:
    uv run python3 compare_communities.py                     # threshold 0.94
    uv run python3 compare_communities.py --threshold 0.9
    uv run python3 compare_communities.py --resolution 1.5    # Louvain resolution
    uv run python3 compare_communities.py --csv comm.csv
"""
import argparse
from collections import Counter

import numpy as np
import pandas as pd
import networkx as nx
from networkx.algorithms.community import louvain_communities, modularity
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
    davies_bouldin_score,
)
from sklearn.metrics.pairwise import cosine_similarity

EMBEDDINGS_FILE    = "specter2_embeddings.npy"
EMBEDDINGS_FT_FILE = "specter2finetuned_embeddings.npy"


def build_graph(embeddings: np.ndarray, threshold: float) -> nx.Graph:
    sim = cosine_similarity(embeddings).astype(np.float32)
    np.fill_diagonal(sim, 0.0)
    iu, ju = np.triu_indices(len(embeddings), k=1)
    sims = sim[iu, ju]
    mask = sims >= threshold
    G = nx.Graph()
    G.add_nodes_from(range(len(embeddings)))
    G.add_weighted_edges_from(
        zip(iu[mask].tolist(), ju[mask].tolist(), sims[mask].astype(float).tolist())
    )
    return G


def louvain_partition(G: nx.Graph, resolution: float, seed: int):
    """Return (label_array, list_of_communities_as_sets, modularity)."""
    n = G.number_of_nodes()
    if G.number_of_edges() == 0:
        labels = np.arange(n)   # every node its own "community"
        return labels, [{i} for i in range(n)], float("nan")

    communities = louvain_communities(G, weight="weight",
                                      resolution=resolution, seed=seed)
    communities = sorted(communities, key=len, reverse=True)
    labels = np.full(n, -1, dtype=np.int64)
    for cid, comm in enumerate(communities):
        for node in comm:
            labels[node] = cid
    mod = modularity(G, communities, weight="weight", resolution=resolution)
    return labels, communities, mod


def per_graph_summary(name: str, labels: np.ndarray,
                      communities: list, mod: float) -> dict:
    sizes = sorted((len(c) for c in communities), reverse=True)
    singletons = sum(1 for s in sizes if s == 1)
    non_singletons = [s for s in sizes if s > 1]
    return {
        "graph": name,
        "num_communities": len(communities),
        "non_singleton_communities": len(non_singletons),
        "singletons": singletons,
        "largest_community": sizes[0] if sizes else 0,
        "median_nonsingleton_size": int(np.median(non_singletons)) if non_singletons else 0,
        "modularity": mod,
    }


def embedding_quality(embeddings: np.ndarray, labels: np.ndarray) -> dict:
    """Silhouette (cosine) & Davies-Bouldin on embeddings, keeping only
    nodes whose community has ≥2 members."""
    counts = Counter(labels.tolist())
    keep_labels = {lbl for lbl, c in counts.items() if c >= 2}
    mask = np.array([lbl in keep_labels for lbl in labels])
    if mask.sum() < 2 or len({labels[i] for i in np.where(mask)[0]}) < 2:
        return {"silhouette_cosine": float("nan"),
                "davies_bouldin": float("nan"),
                "nodes_scored": int(mask.sum())}
    X = embeddings[mask]
    y = labels[mask]
    sil = float(silhouette_score(X, y, metric="cosine"))
    db  = float(davies_bouldin_score(X, y))
    return {"silhouette_cosine": sil,
            "davies_bouldin": db,
            "nodes_scored": int(mask.sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.94,
                    help="Cosine-similarity threshold for edge inclusion (default 0.94)")
    ap.add_argument("--resolution", type=float, default=1.0,
                    help="Louvain resolution parameter (default 1.0)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--csv", type=str, default=None,
                    help="Optional CSV output path for the summary table")
    args = ap.parse_args()

    print("Loading embeddings…")
    emb_orig = np.load(EMBEDDINGS_FILE).astype(np.float32)
    emb_ft   = np.load(EMBEDDINGS_FT_FILE).astype(np.float32)
    assert emb_orig.shape[0] == emb_ft.shape[0], "node count mismatch"
    n = emb_orig.shape[0]

    print(f"Building graphs at threshold {args.threshold}…")
    G_o = build_graph(emb_orig, args.threshold)
    G_f = build_graph(emb_ft,   args.threshold)
    print(f"  original:   {G_o.number_of_edges()} edges")
    print(f"  fine-tuned: {G_f.number_of_edges()} edges")

    print(f"Running Louvain (resolution={args.resolution}, seed={args.seed})…")
    lab_o, comm_o, mod_o = louvain_partition(G_o, args.resolution, args.seed)
    lab_f, comm_f, mod_f = louvain_partition(G_f, args.resolution, args.seed)

    sum_o = per_graph_summary("original",   lab_o, comm_o, mod_o)
    sum_f = per_graph_summary("fine-tuned", lab_f, comm_f, mod_f)

    print("\nComputing silhouette / Davies-Bouldin on embeddings…")
    qual_o = embedding_quality(emb_orig, lab_o)
    qual_f = embedding_quality(emb_ft,   lab_f)
    sum_o.update(qual_o)
    sum_f.update(qual_f)

    df = pd.DataFrame([sum_o, sum_f]).set_index("graph").T
    df["delta (ft - orig)"] = pd.to_numeric(df["fine-tuned"], errors="coerce") \
                            - pd.to_numeric(df["original"],   errors="coerce")

    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    pd.set_option("display.max_rows", None)
    print("\n=== Per-graph community structure ===")
    print(df)

    print("\n=== Cross-graph partition agreement (same nodes, both Louvain) ===")
    nmi = normalized_mutual_info_score(lab_o, lab_f)
    ari = adjusted_rand_score(lab_o, lab_f)
    print(f"  NMI (normalized mutual information): {nmi:.4f}")
    print(f"  ARI (adjusted Rand index):           {ari:.4f}")

    # top community size distribution (helps see concentration)
    top_k = 8
    print(f"\n=== Top-{top_k} community sizes ===")
    o_sizes = sorted((len(c) for c in comm_o), reverse=True)[:top_k]
    f_sizes = sorted((len(c) for c in comm_f), reverse=True)[:top_k]
    print(f"  original:   {o_sizes}")
    print(f"  fine-tuned: {f_sizes}")

    if args.csv:
        df.to_csv(args.csv)
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()
