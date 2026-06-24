"""
Fixed-density comparison of the four paper graphs.

For each method —
    Coupling   (shared-reference counts from OpenAlex)
    TF-IDF     (cosine similarity, bag-of-words TF-IDF)
    SPECTER2   (cosine similarity, original embeddings)
    Fine-tuned (cosine similarity, LoRA-tuned embeddings)
— a graph is built with the SAME number of edges by keeping the top-weighted
edges globally, at several target densities (default 1, 2, 5, 10 %).

This isolates the effect of edge weighting from the effect of graph density:
it answers "are the fine-tuned gains still present when every method has the
same number of edges?", which the single-threshold table cannot.

For each (method, density) we report:
    target / actual edges, isolated nodes, connected components,
    largest connected component, Louvain modularity, silhouette,
    Davies--Bouldin.

Silhouette / Davies--Bouldin score each graph's Louvain communities in a cosine
vector space: each embedding graph in its own space, the coupling graph in the
original SPECTER2 space (it has no space of its own). Node i of the coupling
matrix is aligned with embedding row i, as in the interactive viewer; the few
papers without coupling data appear as isolated nodes.

Outputs: fixed_density_comparison.csv / .txt / .tex / .png

Usage:
    uv run python3 compare_fixed_density.py
    uv run python3 compare_fixed_density.py --densities 0.01 0.02 0.05 0.10
"""
import argparse
from collections import Counter

import numpy as np
import pandas as pd
import scipy.sparse as sp
import networkx as nx
from networkx.algorithms.community import louvain_communities, modularity as nx_modularity
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.metrics.pairwise import cosine_similarity

EMBEDDINGS_FILE    = "specter2_embeddings.npy"
EMBEDDINGS_FT_FILE = "specter2finetuned_embeddings.npy"
TFIDF_FILE         = "tfidf_embeddings.npy"
COUPLING_FILE      = "ArtCon_coupling.npz"

NA = "--"


# ────────────────── ranked edge lists (computed once per method) ──────────────────
def ranked_sim_edges(embeddings, n_nodes):
    """Upper-triangle (i<j) cosine-similarity edges, sorted by weight descending.
    Returns (rows, cols, weights) as parallel arrays."""
    sim = cosine_similarity(embeddings).astype(np.float32)
    np.fill_diagonal(sim, 0.0)
    iu, ju = np.triu_indices(n_nodes, k=1)
    w = sim[iu, ju]
    order = np.argsort(w)[::-1]           # descending
    return iu[order], ju[order], w[order]


def ranked_coupling_edges(matrix):
    """Upper-triangle (i<j) coupling edges, sorted by shared-reference count
    descending. Returns (rows, cols, weights)."""
    coo = matrix.tocoo()
    keep = coo.row < coo.col
    r, c, w = coo.row[keep], coo.col[keep], coo.data[keep].astype(np.float64)
    order = np.argsort(w, kind="stable")[::-1]   # descending; stable for tie reproducibility
    return r[order], c[order], w[order]


def top_k_graph(rows, cols, weights, k, n_nodes):
    """Graph on n_nodes keeping the k top-weighted edges (or all, if fewer)."""
    k = min(k, len(weights))
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    G.add_weighted_edges_from(
        zip(rows[:k].tolist(), cols[:k].tolist(), weights[:k].astype(float).tolist())
    )
    return G


# ────────────────────────── metrics ──────────────────────────
def louvain_labels(G, resolution, seed):
    """(label_array, communities, modularity). Edgeless -> all singletons."""
    n = G.number_of_nodes()
    if G.number_of_edges() == 0:
        return np.arange(n), [{i} for i in range(n)], float("nan")
    comms = louvain_communities(G, weight="weight", resolution=resolution, seed=seed)
    comms = sorted(comms, key=len, reverse=True)
    labels = np.full(n, -1, dtype=np.int64)
    for cid, comm in enumerate(comms):
        for node in comm:
            labels[node] = cid
    mod = float(nx_modularity(G, comms, weight="weight", resolution=resolution))
    return labels, comms, mod


def embedding_quality(embeddings, labels):
    """Silhouette (cosine) & Davies-Bouldin on `embeddings`, keeping only nodes
    whose community has >= 2 members."""
    counts = Counter(labels.tolist())
    keep = {lbl for lbl, c in counts.items() if c >= 2}
    mask = np.array([lbl in keep for lbl in labels])
    if mask.sum() < 2 or len(set(labels[mask].tolist())) < 2:
        return float("nan"), float("nan")
    X, y = embeddings[mask], labels[mask]
    return (float(silhouette_score(X, y, metric="cosine")),
            float(davies_bouldin_score(X, y)))


def structural(G):
    comps = list(nx.connected_components(G))
    return {
        "edges": G.number_of_edges(),
        "isolated": int(sum(1 for _, d in G.degree() if d == 0)),
        "components": len(comps),
        "lcc": max((len(c) for c in comps), default=0),
        "num_communities": 0,   # filled by caller
    }


def metrics_for_graph(G, score_embeddings, resolution, seed, target_edges):
    st = structural(G)
    lab, comms, mod = louvain_labels(G, resolution, seed)
    sil, db = embedding_quality(score_embeddings, lab)
    nontrivial = sum(1 for c in comms if len(c) >= 2)
    return {
        "target_edges": int(target_edges),
        "edges": st["edges"],
        "isolated": st["isolated"],
        "components": st["components"],
        "lcc": st["lcc"],
        "num_communities": nontrivial,
        "modularity": mod,
        "silhouette": sil,
        "davies_bouldin": db,
    }


# ────────────────────────── rendering ──────────────────────────
ROWS = [
    ("target_edges",    "Target edges",                 "int"),
    ("edges",           "Actual edges",                 "int"),
    ("isolated",        "Isolated nodes",               "int"),
    ("components",      "Connected components",         "int"),
    ("lcc",             "Largest connected component",  "int"),
    ("num_communities", "Louvain communities (>=2)",    "int"),
    ("modularity",      "Modularity",                   "f4"),
    ("silhouette",      "Silhouette",                   "f4"),
    ("davies_bouldin",  "Davies--Bouldin",              "f4"),
]
COLS = [("coupling", "Coupling"), ("tfidf", "TF-IDF"),
        ("emb", "SPECTER2"), ("ft", "Fine-tuned")]


def fmt(value, kind):
    if isinstance(value, str):
        return value
    if isinstance(value, float) and np.isnan(value):
        return NA
    return f"{value:.4f}" if kind == "f4" else f"{int(value)}"


def render_text(per_density, n, max_edges):
    W = 12
    out = [f"Fixed-density graph comparison  (n = {n} papers, "
           f"max possible edges = {max_edges})",
           "Top-weighted edges kept globally per method at each target density.",
           ""]
    for d, data in per_density:
        out.append(f"=== Density {d:.0%}  (target {int(round(d*max_edges))} edges) ===")
        head = f"{'Metric':30s}" + "".join(f"{lbl:>{W}s}" for _, lbl in COLS)
        out.append(head)
        out.append("-" * len(head))
        for key, label, kind in ROWS:
            cells = "".join(f"{fmt(data[c][key], kind):>{W}s}" for c, _ in COLS)
            out.append(f"{label.replace('--','–'):30s}{cells}")
        out.append("")
    out += [
        "Notes:",
        "  • Edges are the globally top-weighted pairs (cosine similarity for the",
        "    embedding methods, shared-reference count for coupling).",
        "  • 'Actual edges' < 'Target edges' means the method ran out of non-zero",
        "    edges (the coupling graph has a finite number of shared-reference pairs).",
        "  • Silhouette / Davies–Bouldin score each method's Louvain communities in a",
        "    cosine space: each embedding graph in its own space, coupling in the",
        "    original SPECTER2 space (it has no space of its own).",
    ]
    return "\n".join(out)


def render_latex(per_density, n, max_edges):
    colspec = "l" + "r" * len(COLS)
    out = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Fixed-density comparison of the bibliographic-coupling, TF-IDF, "
        r"SPECTER2 and fine-tuned graphs. At each target density the top-weighted "
        r"edges are kept globally, so all four graphs have the same number of edges "
        r"(up to the coupling graph's finite edge budget). Silhouette and "
        r"Davies--Bouldin score each graph's Louvain communities in a cosine space "
        rf"(coupling in the original SPECTER2 space). $n={n}$ papers, "
        rf"maximum ${max_edges}$ possible edges.}}",
        r"\label{tab:fixed-density}",
        rf"\begin{{tabular}}{{{colspec}}}",
        r"\hline",
        "Metric & " + " & ".join(lbl for _, lbl in COLS) + r" \\",
        r"\hline",
    ]
    for d, data in per_density:
        out.append(r"\multicolumn{%d}{l}{\emph{Density %.0f\%% "
                   r"(target %d edges)}} \\" % (len(COLS) + 1, d * 100,
                                                round(d * max_edges)))
        out.append(r"\hline")
        for key, label, kind in ROWS:
            cells = [fmt(data[c][key], kind) for c, _ in COLS]
            cells = [c if c != NA else r"\textemdash" for c in cells]
            body = " & ".join(f"{c:>12s}" for c in cells)
            tex_label = label.replace(">=", r"$\geq$")
            out.append(f"{tex_label:30s} & {body} \\\\")
        out.append(r"\hline")
    out += [r"\end{tabular}", r"\end{table}"]
    return "\n".join(out)


def render_csv(per_density, max_edges, path):
    recs = []
    for d, data in per_density:
        for c, label in COLS:
            rec = {"method": label, "density": d,
                   "target_edges": int(round(d * max_edges))}
            rec.update({k: data[c][k] for k, _, _ in ROWS})
            recs.append(rec)
    df = pd.DataFrame(recs)
    df.to_csv(path, index=False)
    print(f"Wrote {path}")
    return df


def render_png(per_density, max_edges, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    densities = [d for d, _ in per_density]
    x = [d * 100 for d in densities]
    colors = {"coupling": "#2ca02c", "tfidf": "#9467bd",
              "emb": "#1f77b4", "ft": "#d62728"}
    panels = [("modularity", "Louvain modularity"),
              ("silhouette", "Silhouette (cosine)"),
              ("davies_bouldin", "Davies–Bouldin")]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (key, title) in zip(axes, panels):
        for c, label in COLS:
            y = [per_density[i][1][c][key] for i in range(len(densities))]
            ax.plot(x, y, "-o", label=label, color=colors[c],
                    markersize=5, linewidth=2)
        ax.set_xlabel("target density (%)", fontsize=14)
        ax.set_ylabel(title, fontsize=14)
        ax.set_title(title, fontsize=15)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="both", labelsize=12)
    axes[0].legend(fontsize=12)
    fig.suptitle("Fixed-density graph comparison", fontsize=16)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--densities", type=float, nargs="+",
                    default=[0.01, 0.02, 0.05, 0.10])
    ap.add_argument("--resolution", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--csv", default="fixed_density_comparison.csv")
    ap.add_argument("--txt", default="fixed_density_comparison.txt")
    ap.add_argument("--tex", default="fixed_density_comparison.tex")
    ap.add_argument("--png", default="fixed_density_comparison.png")
    args = ap.parse_args()

    print("Loading embeddings & coupling matrix…")
    emb_o = np.load(EMBEDDINGS_FILE).astype(np.float32)
    emb_f = np.load(EMBEDDINGS_FT_FILE).astype(np.float32)
    emb_t = np.load(TFIDF_FILE).astype(np.float32)
    mat   = sp.load_npz(COUPLING_FILE)
    n = emb_o.shape[0]
    max_edges = n * (n - 1) // 2
    print(f"  embeddings {emb_o.shape}, fine-tuned {emb_f.shape}, "
          f"tf-idf {emb_t.shape}, coupling {mat.shape} ({mat.nnz} nnz)")
    print(f"  n = {n}, max possible edges = {max_edges}")

    print("Ranking edges per method (once)…")
    ranked = {
        "coupling": (ranked_coupling_edges(mat), emb_o),     # scored in SPECTER2 space
        "tfidf":    (ranked_sim_edges(emb_t, n), emb_t),
        "emb":      (ranked_sim_edges(emb_o, n), emb_o),
        "ft":       (ranked_sim_edges(emb_f, n), emb_f),
    }
    for c, _ in COLS:
        (r, cc, w), _ = ranked[c]
        print(f"  [{c}] {len(w)} candidate edges (max weight {w.max():.4g})")

    per_density = []
    for d in sorted(args.densities):
        k = int(round(d * max_edges))
        print(f"\nDensity {d:.0%}  → top {k} edges")
        data = {}
        for c, label in COLS:
            (rows, cols, w), space = ranked[c]
            G = top_k_graph(rows, cols, w, k, n)
            data[c] = metrics_for_graph(G, space, args.resolution, args.seed, k)
            cap = "  (CAPPED: ran out of edges)" if data[c]["edges"] < k else ""
            print(f"  [{label:10s}] edges={data[c]['edges']:6d}  "
                  f"comp={data[c]['components']:4d}  lcc={data[c]['lcc']:4d}  "
                  f"Q={data[c]['modularity']:.4f}  sil={data[c]['silhouette']:.4f}  "
                  f"DB={data[c]['davies_bouldin']:.4f}{cap}")
        per_density.append((d, data))

    text = render_text(per_density, n, max_edges)
    print("\n" + text + "\n")
    with open(args.txt, "w") as f:
        f.write(text + "\n")
    print(f"Wrote {args.txt}")
    with open(args.tex, "w") as f:
        f.write(render_latex(per_density, n, max_edges) + "\n")
    print(f"Wrote {args.tex}")
    render_csv(per_density, max_edges, args.csv)
    render_png(per_density, max_edges, args.png)


if __name__ == "__main__":
    main()
