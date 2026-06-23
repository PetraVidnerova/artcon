"""
Summary comparison table for the paper graphs:

    1. Bibliographic-coupling graph   (shared-reference counts from OpenAlex)
    2. TF-IDF lexical graph           (cosine similarity, bag-of-words TF-IDF)
    3. SPECTER2 embedding graph       (cosine similarity, original embeddings)
    4. Fine-tuned embedding graph     (cosine similarity, LoRA-tuned embeddings)

Mirrors the original two-column (Original vs. Fine-tuned) summary table, adding
the coupling and TF-IDF graphs as further columns. (Filename kept for history;
it now compares four graphs.)

Row semantics
-------------
"Complete graph" rows (mean/std cosine similarity, silhouette, Davies-Bouldin
over Louvain communities of the *complete* similarity graph) require a cosine
vector space, so they are N/A ("--") for the coupling graph.

"Thresholded graph" rows are structural and apply to all three:
  - embedding graphs are thresholded at cosine similarity >= 0.94
  - the coupling graph is thresholded at >= 2 shared references
Silhouette / Davies-Bouldin of the thresholded-graph Louvain communities are
scored in each graph's embedding space; the coupling communities are scored in
the *original* SPECTER2 embedding space (the coupling graph has no space of its
own).

Usage:
    uv run python3 compare_three_graphs.py
    uv run python3 compare_three_graphs.py --sim-threshold 0.94 --coupling-min 2
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


# ────────────────────────── graph builders ──────────────────────────
def sim_graph(embeddings, threshold, n_nodes):
    sim = cosine_similarity(embeddings).astype(np.float32)
    np.fill_diagonal(sim, 0.0)
    iu, ju = np.triu_indices(n_nodes, k=1)
    sims = sim[iu, ju]
    mask = sims >= threshold
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    G.add_weighted_edges_from(
        zip(iu[mask].tolist(), ju[mask].tolist(), sims[mask].astype(float).tolist())
    )
    return G


def coupling_graph(matrix, min_shared, n_nodes):
    coo = matrix.tocoo()
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    for i, j, s in zip(coo.row, coo.col, coo.data):
        if i < j and s >= min_shared:
            G.add_edge(int(i), int(j), weight=float(s))
    return G


# ────────────────────────── metrics ──────────────────────────
def louvain_labels(G, resolution, seed):
    """Return (label_array, communities, modularity). Edgeless -> all singletons."""
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
    n = G.number_of_nodes()
    comps = list(nx.connected_components(G))
    lcc = max((len(c) for c in comps), default=0)
    return {
        "edges": G.number_of_edges(),
        "isolated": int(sum(1 for _, d in G.degree() if d == 0)),
        "components": len(comps),
        "lcc": lcc,
    }


def column_for_sim(name, embeddings, sim_threshold, resolution, seed):
    """Full metric column for an embedding-based graph."""
    n = embeddings.shape[0]
    print(f"  [{name}] complete-graph similarity stats + Louvain…")
    sim = cosine_similarity(embeddings).astype(np.float32)
    np.fill_diagonal(sim, 0.0)
    iu, ju = np.triu_indices(n, k=1)
    sims_all = sim[iu, ju]

    # Complete (all-pairs) graph -> Louvain -> embedding quality
    Gc = sim_graph(embeddings, threshold=-1.0, n_nodes=n)   # all pairs (cosine>0 here)
    lab_c, _, _ = louvain_labels(Gc, resolution, seed)
    sil_c, db_c = embedding_quality(embeddings, lab_c)

    # Thresholded graph
    print(f"  [{name}] thresholded graph (>= {sim_threshold}) + Louvain…")
    Gt = sim_graph(embeddings, threshold=sim_threshold, n_nodes=n)
    st = structural(Gt)
    lab_t, _, mod_t = louvain_labels(Gt, resolution, seed)
    sil_t, db_t = embedding_quality(embeddings, lab_t)

    return {
        "sim_mean": sims_all.mean(),
        "sim_std":  sims_all.std(),
        "sil_complete": sil_c,
        "db_complete":  db_c,
        "edges": st["edges"],
        "isolated": st["isolated"],
        "components": st["components"],
        "lcc": st["lcc"],
        "modularity": mod_t,
        "sil_thr": sil_t,
        "db_thr":  db_t,
    }


def column_for_coupling(matrix, min_shared, score_embeddings, resolution, seed):
    """Metric column for the coupling graph; quality scored in `score_embeddings`."""
    n = score_embeddings.shape[0]   # use full node set (matches export_html viewer)
    print(f"  [coupling] thresholded graph (>= {min_shared} shared refs) + Louvain…")
    G = coupling_graph(matrix, min_shared, n_nodes=n)
    st = structural(G)
    lab, _, mod = louvain_labels(G, resolution, seed)
    sil, db = embedding_quality(score_embeddings, lab)
    return {
        "sim_mean": NA,
        "sim_std":  NA,
        "sil_complete": NA,
        "db_complete":  NA,
        "edges": st["edges"],
        "isolated": st["isolated"],
        "components": st["components"],
        "lcc": st["lcc"],
        "modularity": mod,
        "sil_thr": sil,
        "db_thr":  db,
    }


# ────────────────────────── rendering ──────────────────────────
# (row key, human label, kind)  kind: f4=4-decimal float, int=integer
ROWS = [
    ("sim_mean",     "Mean similarity (complete graph)",        "f4"),
    ("sim_std",      "Similarity std. dev. (complete graph)",   "f4"),
    ("sil_complete", "Silhouette (complete graph)",             "f4"),
    ("db_complete",  "Davies--Bouldin (complete graph)",        "f4"),
    ("edges",        "Edges (thresholded)",                     "int"),
    ("isolated",     "Isolated nodes (thresholded)",            "int"),
    ("components",   "Connected components (thresholded)",      "int"),
    ("lcc",          "Largest connected component (thresholded)","int"),
    ("modularity",   "Modularity (thresholded)",                "f4"),
    ("sil_thr",      "Silhouette (thresholded)",                "f4"),
    ("db_thr",       "Davies--Bouldin (thresholded)",           "f4"),
]
COLS = [("coupling", "Coupling"), ("tfidf", "TF-IDF"),
        ("emb", "Embeddings"), ("ft", "Fine-tuned")]


def fmt(value, kind):
    if isinstance(value, str):           # NA marker
        return value
    if isinstance(value, float) and np.isnan(value):
        return NA
    return f"{value:.4f}" if kind == "f4" else f"{int(value)}"


def render_text(data, sim_threshold, coupling_min, tfidf_threshold):
    W = 12
    head = f"{'Metric':42s}" + "".join(f"{lbl:>{W}s}" for _, lbl in COLS)
    lines = [head, "-" * len(head)]
    for key, label, kind in ROWS:
        cells = "".join(f"{fmt(data[c][key], kind):>{W}s}" for c, _ in COLS)
        lines.append(f"{label.replace('--','–'):42s}{cells}")
    lines += ["",
              f"Thresholds: embeddings/fine-tuned at cosine similarity >= {sim_threshold}; "
              f"TF-IDF at cosine similarity >= {tfidf_threshold}; "
              f"coupling at >= {coupling_min} shared references.",
              "'--' = not applicable (coupling graph has no cosine-similarity space).",
              "Note: coupling covers 696 papers vs. 704 embeddings; coupling node i is aligned",
              "with embedding row i (as in the viewer), so the 8 papers without coupling data",
              "appear as isolated nodes in the coupling column."]
    return "\n".join(lines)


def render_latex(data, sim_threshold, coupling_min, tfidf_threshold):
    colspec = "l" + "r" * len(COLS)
    header = "Metric & " + " & ".join(lbl for _, lbl in COLS) + r" \\"
    out = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Structural and clustering-quality comparison of the bibliographic-coupling "
        r"graph, the lexical TF-IDF graph, the SPECTER2 embedding graph, and the fine-tuned "
        r"embedding graph. "
        rf"Embedding graphs are thresholded at cosine similarity $\geq {sim_threshold}$; "
        rf"the TF-IDF graph at cosine similarity $\geq {tfidf_threshold}$; "
        rf"the coupling graph at $\geq {coupling_min}$ shared references. "
        r"`Complete graph' rows require a cosine vector space and are not applicable to the "
        r"coupling graph. Silhouette and Davies--Bouldin score each graph's Louvain communities "
        r"in its embedding space (the coupling communities in the original SPECTER2 space). "
        r"Note: the coupling matrix covers 696 papers while the embeddings cover 704; "
        r"coupling node $i$ is aligned with embedding row $i$ (as in the interactive viewer), "
        r"so the 8 papers without coupling data appear as isolated nodes in the coupling column.}",
        r"\label{tab:graph-comparison}",
        rf"\begin{{tabular}}{{{colspec}}}",
        r"\hline",
        header,
        r"\hline",
    ]
    for key, label, kind in ROWS:
        cells = [fmt(data[c][key], kind) for c, _ in COLS]
        cells = [c if c != NA else r"\textemdash" for c in cells]
        body = " & ".join(f"{c:>12s}" for c in cells)
        out.append(f"{label:42s} & {body} \\\\")
    out += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(out)


def render_png(data, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cell_text, row_labels = [], []
    for key, label, kind in ROWS:
        row_labels.append(label.replace("--", "–"))
        cell_text.append([fmt(data[c][key], kind).replace(NA, "—") for c, _ in COLS])

    fig, ax = plt.subplots(figsize=(8.5, 0.45 * len(ROWS) + 0.6))
    ax.axis("off")
    tbl = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=[lbl for _, lbl in COLS],
        cellLoc="center", rowLoc="left", loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 1.4)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("none")
        if r == 0:                       # header row
            cell.set_text_props(fontweight="bold")
            cell.visible_edges = "B"
        if c == -1:                      # row-label column
            cell.set_text_props(ha="left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-threshold", type=float, default=0.94)
    ap.add_argument("--tfidf-threshold", type=float, default=0.10,
                    help="Cosine threshold for the TF-IDF graph (default 0.10; "
                         "TF-IDF cosines are much smaller than SPECTER2's)")
    ap.add_argument("--coupling-min", type=int, default=2)
    ap.add_argument("--resolution", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tex", default="three_graph_comparison.tex")
    ap.add_argument("--txt", default="three_graph_comparison.txt")
    ap.add_argument("--png", default="three_graph_comparison.png")
    args = ap.parse_args()

    print("Loading embeddings & coupling matrix…")
    emb_o = np.load(EMBEDDINGS_FILE).astype(np.float32)
    emb_f = np.load(EMBEDDINGS_FT_FILE).astype(np.float32)
    emb_t = np.load(TFIDF_FILE).astype(np.float32)
    mat   = sp.load_npz(COUPLING_FILE)
    print(f"  embeddings: {emb_o.shape}, fine-tuned: {emb_f.shape}, "
          f"tf-idf: {emb_t.shape}, coupling: {mat.shape} ({mat.nnz} nnz)")

    print("Computing columns…")
    data = {
        "emb":      column_for_sim("embeddings", emb_o, args.sim_threshold, args.resolution, args.seed),
        "ft":       column_for_sim("fine-tuned", emb_f, args.sim_threshold, args.resolution, args.seed),
        "tfidf":    column_for_sim("tf-idf", emb_t, args.tfidf_threshold, args.resolution, args.seed),
        "coupling": column_for_coupling(mat, args.coupling_min, emb_o, args.resolution, args.seed),
    }

    text = render_text(data, args.sim_threshold, args.coupling_min, args.tfidf_threshold)
    print("\n" + text + "\n")

    with open(args.txt, "w") as f:
        f.write(text + "\n")
    print(f"Wrote {args.txt}")
    with open(args.tex, "w") as f:
        f.write(render_latex(data, args.sim_threshold, args.coupling_min, args.tfidf_threshold) + "\n")
    print(f"Wrote {args.tex}")
    render_png(data, args.png)


if __name__ == "__main__":
    main()
