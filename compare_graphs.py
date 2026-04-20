"""
Compare global structural properties of the original vs. fine-tuned
similarity graph built from SPECTER2 embeddings.

Both graphs share the same node set (one node per paper). Edges are
cosine similarities between embeddings; edges below --threshold are dropped.

Usage:
    uv run python3 compare_graphs.py                 # default threshold 0.8
    uv run python3 compare_graphs.py --threshold 0.75
    uv run python3 compare_graphs.py --threshold 0.8 --csv out.csv
"""
import argparse
import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import pearsonr, spearmanr
from scipy.sparse.linalg import eigsh
from sklearn.metrics.pairwise import cosine_similarity

EMBEDDINGS_FILE    = "specter2_embeddings.npy"
EMBEDDINGS_FT_FILE = "specter2finetuned_embeddings.npy"


def build_graph(embeddings: np.ndarray, threshold: float) -> tuple[nx.Graph, np.ndarray]:
    """Return (graph, upper-triangle similarity vector) — sims returned for
    cross-graph weight-correlation regardless of threshold."""
    sim = cosine_similarity(embeddings).astype(np.float32)
    np.fill_diagonal(sim, 0.0)
    iu, ju = np.triu_indices(len(embeddings), k=1)
    sims = sim[iu, ju]

    mask = sims >= threshold
    G = nx.Graph()
    G.add_nodes_from(range(len(embeddings)))
    ei, ej, es = iu[mask], ju[mask], sims[mask]
    G.add_weighted_edges_from(zip(ei.tolist(), ej.tolist(), es.astype(float).tolist()))
    return G, sims


def spectral_gap(G: nx.Graph, k: int = 2) -> float | None:
    """Gap between the two smallest eigenvalues of the normalized Laplacian
    on the largest connected component. Returns None if the LCC is tiny."""
    if G.number_of_edges() == 0:
        return None
    lcc_nodes = max(nx.connected_components(G), key=len)
    if len(lcc_nodes) < 3:
        return None
    H = G.subgraph(lcc_nodes)
    L = nx.normalized_laplacian_matrix(H).astype(float)
    n = L.shape[0]
    try:
        vals = eigsh(L, k=min(k, n - 1), which="SM", return_eigenvectors=False)
        vals = np.sort(vals)
        return float(vals[1] - vals[0]) if len(vals) >= 2 else None
    except Exception:
        return None


def describe(G: nx.Graph, sims_all: np.ndarray, threshold: float, name: str) -> dict:
    n, m = G.number_of_nodes(), G.number_of_edges()
    weights = np.array([d["weight"] for _, _, d in G.edges(data=True)], dtype=float)
    strengths = np.array([s for _, s in G.degree(weight="weight")], dtype=float)
    degrees = np.array([d for _, d in G.degree()], dtype=float)

    components = list(nx.connected_components(G))
    lcc_nodes = max(components, key=len) if components else set()
    H = G.subgraph(lcc_nodes).copy() if lcc_nodes else nx.Graph()

    # Average shortest path on LCC, using distance = 1 - similarity.
    if H.number_of_edges() > 0:
        for u, v, d in H.edges(data=True):
            d["distance"] = max(1e-6, 1.0 - d["weight"])
        try:
            avg_path = nx.average_shortest_path_length(H, weight="distance")
        except Exception:
            avg_path = float("nan")
    else:
        avg_path = float("nan")

    try:
        assort = nx.degree_assortativity_coefficient(G, weight="weight") if m else float("nan")
    except Exception:
        assort = float("nan")

    stats = {
        "graph": name,
        "threshold": threshold,
        "nodes": n,
        "edges": m,
        "density": nx.density(G),
        "isolated_nodes": int(sum(1 for _, d in G.degree() if d == 0)),
        "num_components": len(components),
        "lcc_size": len(lcc_nodes),
        "lcc_fraction": len(lcc_nodes) / n if n else float("nan"),
        # edge-weight distribution (over retained edges)
        "w_mean": float(weights.mean()) if m else float("nan"),
        "w_std":  float(weights.std())  if m else float("nan"),
        "w_min":  float(weights.min())  if m else float("nan"),
        "w_q25":  float(np.quantile(weights, 0.25)) if m else float("nan"),
        "w_median": float(np.median(weights)) if m else float("nan"),
        "w_q75":  float(np.quantile(weights, 0.75)) if m else float("nan"),
        "w_max":  float(weights.max())  if m else float("nan"),
        # full-distribution stats over all pairs (unfiltered)
        "sim_all_mean":   float(sims_all.mean()),
        "sim_all_std":    float(sims_all.std()),
        "sim_all_median": float(np.median(sims_all)),
        # degree / strength
        "deg_mean":    float(degrees.mean()),
        "deg_std":     float(degrees.std()),
        "deg_max":     float(degrees.max()) if n else float("nan"),
        "strength_mean": float(strengths.mean()),
        "strength_std":  float(strengths.std()),
        # clustering
        "avg_clustering_unweighted": nx.average_clustering(G) if m else 0.0,
        "avg_clustering_weighted":   nx.average_clustering(G, weight="weight") if m else 0.0,
        "transitivity": nx.transitivity(G) if m else 0.0,
        # paths / mixing
        "avg_shortest_path_lcc": avg_path,
        "degree_assortativity_w": assort,
        # spectrum
        "spectral_gap_lcc": spectral_gap(G),
    }
    return stats


def cross_graph_edge_correlation(sims_a: np.ndarray, sims_b: np.ndarray) -> dict:
    p, _ = pearsonr(sims_a, sims_b)
    s, _ = spearmanr(sims_a, sims_b)
    diff = sims_b - sims_a
    return {
        "pearson_edge_weight": float(p),
        "spearman_edge_weight": float(s),
        "frobenius_diff": float(np.linalg.norm(diff) * np.sqrt(2)),  # upper tri → full
        "mean_weight_shift": float(diff.mean()),
        "std_weight_shift":  float(diff.std()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.8,
                    help="Minimum cosine similarity for an edge (default 0.8)")
    ap.add_argument("--csv", type=str, default=None,
                    help="Optional path to write the comparison table as CSV")
    args = ap.parse_args()

    print(f"Loading embeddings…")
    emb_orig = np.load(EMBEDDINGS_FILE).astype(np.float32)
    emb_ft   = np.load(EMBEDDINGS_FT_FILE).astype(np.float32)
    assert emb_orig.shape[0] == emb_ft.shape[0], "node count mismatch"

    print(f"Building graphs at threshold {args.threshold}…")
    G_orig, sims_orig = build_graph(emb_orig, args.threshold)
    G_ft,   sims_ft   = build_graph(emb_ft,   args.threshold)

    print("Computing global properties (this can take a minute)…")
    s_orig = describe(G_orig, sims_orig, args.threshold, "original")
    s_ft   = describe(G_ft,   sims_ft,   args.threshold, "fine-tuned")

    df = pd.DataFrame([s_orig, s_ft]).set_index("graph").T
    df["delta (ft - orig)"] = pd.to_numeric(df["fine-tuned"], errors="coerce") \
                            - pd.to_numeric(df["original"],  errors="coerce")

    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    pd.set_option("display.max_rows", None)
    print("\n=== Global structural properties ===")
    print(df)

    print("\n=== Cross-graph edge-weight comparison (over all pairs) ===")
    cross = cross_graph_edge_correlation(sims_orig, sims_ft)
    for k, v in cross.items():
        print(f"  {k:28s} {v:.4f}")

    if args.csv:
        df.to_csv(args.csv)
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()
