"""
Symmetric k-NN graph comparison of the four paper graphs.

For each method —
    Coupling   (shared-reference counts from OpenAlex)
    TF-IDF     (cosine similarity, bag-of-words TF-IDF)
    SPECTER2   (cosine similarity, original embeddings)
    Fine-tuned (cosine similarity, LoRA-tuned embeddings)
— a symmetric k-NN graph is built: every document is connected to its top-k most
similar documents, and the resulting directed neighbourhoods are symmetrised by
union (i~j if i is in j's top-k OR j is in i's top-k).

Unlike the fixed-density comparison, which keeps a single global set of top-weighted
edges, k-NN gives EVERY node the same neighbourhood budget regardless of where it
sits in the similarity distribution. This is the cleanest cross-method experiment
because literature maps are local-neighbourhood structures, and k-NN sidesteps the
incompatible raw-similarity scales of the four methods (cosine in [-1,1] vs.
integer shared-reference counts): each method only has to rank its own neighbours.

For each (method, k) we report:
    edges, mean degree, isolated nodes, connected components,
    largest connected component, Louvain communities, modularity,
    silhouette, Davies--Bouldin.

Silhouette / Davies--Bouldin score each graph's Louvain communities in a cosine
vector space: each embedding graph in its own space, the coupling graph in the
original SPECTER2 space (it has no space of its own). Node i of the coupling
matrix is aligned with embedding row i, as in the interactive viewer.

A neighbour is only kept if its similarity is strictly positive, so the coupling
graph (whose "similarity" is a shared-reference count) connects a document only to
documents it actually shares references with; documents with fewer than k non-zero
coupling neighbours therefore get a smaller neighbourhood. For the cosine methods
this almost never binds.

Outputs: knn_comparison.csv / .txt / .tex / .png

Usage:
    uv run python3 compare_knn.py
    uv run python3 compare_knn.py --ks 5 10 15 20 30
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


# ────────────────── similarity matrices (computed once per method) ──────────────────
def cosine_matrix(embeddings):
    """Dense cosine-similarity matrix with the diagonal zeroed."""
    sim = cosine_similarity(embeddings).astype(np.float32)
    np.fill_diagonal(sim, 0.0)
    return sim


def coupling_matrix(matrix, n_nodes):
    """Dense shared-reference-count matrix (symmetric) with diagonal zeroed."""
    sim = np.asarray(matrix.todense(), dtype=np.float32)
    # guard against a non-square / smaller matrix; align top-left block on node i
    out = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    m = min(n_nodes, sim.shape[0])
    out[:m, :m] = sim[:m, :m]
    np.fill_diagonal(out, 0.0)
    return out


# ────────────────── symmetric (union) k-NN graph ──────────────────
def knn_graph(sim, k, n_nodes):
    """Symmetric k-NN graph: connect each node to its top-k most similar nodes,
    then symmetrise by union. Only strictly-positive similarities are eligible
    (so zero-coupling pairs are never neighbours). Returns a weighted nx.Graph;
    edge weight is the (symmetric) similarity."""
    k = min(k, n_nodes - 1)
    # top-k columns per row by descending similarity (unordered within the block)
    part = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]

    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    for i in range(n_nodes):
        for j in part[i]:
            w = float(sim[i, j])
            if w <= 0.0:          # not a real neighbour (e.g. zero shared references)
                continue
            a, b = (i, int(j)) if i < j else (int(j), i)
            G.add_edge(a, b, weight=w)   # union symmetrisation: edge kept if either side picked it
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
    degrees = [d for _, d in G.degree()]
    return {
        "edges": G.number_of_edges(),
        "mean_degree": float(np.mean(degrees)) if degrees else 0.0,
        "isolated": int(sum(1 for d in degrees if d == 0)),
        "components": len(comps),
        "lcc": max((len(c) for c in comps), default=0),
    }


def metrics_for_graph(G, score_embeddings, resolution, seed):
    st = structural(G)
    lab, comms, mod = louvain_labels(G, resolution, seed)
    sil, db = embedding_quality(score_embeddings, lab)
    nontrivial = sum(1 for c in comms if len(c) >= 2)
    return {
        "edges": st["edges"],
        "mean_degree": st["mean_degree"],
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
    ("edges",           "Edges",                        "int"),
    ("mean_degree",     "Mean degree",                  "f2"),
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
    if kind == "f4":
        return f"{value:.4f}"
    if kind == "f2":
        return f"{value:.2f}"
    return f"{int(value)}"


def render_text(per_k, n):
    W = 12
    out = [f"Symmetric k-NN graph comparison  (n = {n} papers)",
           "Each node connected to its top-k most similar; symmetrised by union.",
           ""]
    for k, data in per_k:
        out.append(f"=== k = {k} ===")
        head = f"{'Metric':30s}" + "".join(f"{lbl:>{W}s}" for _, lbl in COLS)
        out.append(head)
        out.append("-" * len(head))
        for key, label, kind in ROWS:
            cells = "".join(f"{fmt(data[c][key], kind):>{W}s}" for c, _ in COLS)
            out.append(f"{label.replace('--','–'):30s}{cells}")
        out.append("")
    out += [
        "Notes:",
        "  • Each document is linked to its top-k most similar documents (cosine",
        "    similarity for the embedding methods, shared-reference count for",
        "    coupling); neighbourhoods are symmetrised by union.",
        "  • Only strictly-positive similarities count as neighbours, so the coupling",
        "    graph links a document only to those it shares references with — documents",
        "    with < k non-zero coupling neighbours get a smaller neighbourhood.",
        "  • Mean degree exceeds k when the union adds reciprocal-but-unchosen links.",
        "  • Silhouette / Davies–Bouldin score each method's Louvain communities in a",
        "    cosine space: each embedding graph in its own space, coupling in the",
        "    original SPECTER2 space (it has no space of its own).",
    ]
    return "\n".join(out)


def render_latex(per_k, n):
    colspec = "l" + "r" * len(COLS)
    out = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Symmetric $k$-NN comparison of the bibliographic-coupling, "
        r"TF-IDF, SPECTER2 and fine-tuned graphs. Every document is connected to "
        r"its $k$ most similar documents and the neighbourhoods are symmetrised by "
        r"union, giving each node the same neighbourhood budget irrespective of the "
        r"incompatible raw-similarity scales. Silhouette and Davies--Bouldin score "
        r"each graph's Louvain communities in a cosine space (coupling in the "
        rf"original SPECTER2 space). $n={n}$ papers.}}",
        r"\label{tab:knn}",
        rf"\begin{{tabular}}{{{colspec}}}",
        r"\hline",
        "Metric & " + " & ".join(lbl for _, lbl in COLS) + r" \\",
        r"\hline",
    ]
    for k, data in per_k:
        out.append(r"\multicolumn{%d}{l}{\emph{$k = %d$}} \\" % (len(COLS) + 1, k))
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


def render_csv(per_k, path):
    recs = []
    for k, data in per_k:
        for c, label in COLS:
            rec = {"method": label, "k": k}
            rec.update({key: data[c][key] for key, _, _ in ROWS})
            recs.append(rec)
    df = pd.DataFrame(recs)
    df.to_csv(path, index=False)
    print(f"Wrote {path}")
    return df


def render_png(per_k, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ks = [k for k, _ in per_k]
    colors = {"coupling": "#2ca02c", "tfidf": "#9467bd",
              "emb": "#1f77b4", "ft": "#d62728"}
    panels = [("modularity", "Louvain modularity"),
              ("silhouette", "Silhouette (cosine)"),
              ("davies_bouldin", "Davies–Bouldin")]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (key, title) in zip(axes, panels):
        for c, label in COLS:
            y = [per_k[i][1][c][key] for i in range(len(ks))]
            ax.plot(ks, y, "-o", label=label, color=colors[c],
                    markersize=5, linewidth=2)
        ax.set_xlabel("k (neighbours per node)", fontsize=14)
        ax.set_ylabel(title, fontsize=14)
        ax.set_title(title, fontsize=15)
        ax.set_xticks(ks)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="both", labelsize=12)
    axes[0].legend(fontsize=12)
    fig.suptitle("Symmetric k-NN graph comparison", fontsize=16)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", type=int, nargs="+", default=[5, 10, 15, 20, 30])
    ap.add_argument("--resolution", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--csv", default="knn_comparison.csv")
    ap.add_argument("--txt", default="knn_comparison.txt")
    ap.add_argument("--tex", default="knn_comparison.tex")
    ap.add_argument("--png", default="knn_comparison.png")
    args = ap.parse_args()

    print("Loading embeddings & coupling matrix…")
    emb_o = np.load(EMBEDDINGS_FILE).astype(np.float32)
    emb_f = np.load(EMBEDDINGS_FT_FILE).astype(np.float32)
    emb_t = np.load(TFIDF_FILE).astype(np.float32)
    mat   = sp.load_npz(COUPLING_FILE)
    n = emb_o.shape[0]
    print(f"  embeddings {emb_o.shape}, fine-tuned {emb_f.shape}, "
          f"tf-idf {emb_t.shape}, coupling {mat.shape} ({mat.nnz} nnz)")

    print("Building similarity matrices (once)…")
    sims = {
        "coupling": (coupling_matrix(mat, n), emb_o),     # scored in SPECTER2 space
        "tfidf":    (cosine_matrix(emb_t), emb_t),
        "emb":      (cosine_matrix(emb_o), emb_o),
        "ft":       (cosine_matrix(emb_f), emb_f),
    }

    per_k = []
    for k in sorted(args.ks):
        print(f"\nk = {k}")
        data = {}
        for c, label in COLS:
            sim, space = sims[c]
            G = knn_graph(sim, k, n)
            data[c] = metrics_for_graph(G, space, args.resolution, args.seed)
            print(f"  [{label:10s}] edges={data[c]['edges']:6d}  "
                  f"<deg>={data[c]['mean_degree']:5.2f}  "
                  f"comp={data[c]['components']:4d}  lcc={data[c]['lcc']:4d}  "
                  f"Q={data[c]['modularity']:.4f}  sil={data[c]['silhouette']:.4f}  "
                  f"DB={data[c]['davies_bouldin']:.4f}")
        per_k.append((k, data))

    text = render_text(per_k, n)
    print("\n" + text + "\n")
    with open(args.txt, "w") as f:
        f.write(text + "\n")
    print(f"Wrote {args.txt}")
    with open(args.tex, "w") as f:
        f.write(render_latex(per_k, n) + "\n")
    print(f"Wrote {args.tex}")
    render_csv(per_k, args.csv)
    render_png(per_k, args.png)


if __name__ == "__main__":
    main()
