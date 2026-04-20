"""
Node-level comparison of original vs. fine-tuned SPECTER2 similarity graph.

Produces:
  - node_centrality.csv             — per-paper centralities + ranks + kNN Jaccard
  - node_centrality_summary.txt     — Spearman/Kendall per measure + kNN stats
  - node_centrality_scatter.png     — 2x2 scatter (orig vs ft) per measure
  - knn_overlap_vs_k.png            — mean kNN Jaccard vs k
  - node_centrality.tex             — LaTeX tables (summary, top-shifted papers, kNN sweep)

Usage:
    uv run python3 compare_nodes.py
    uv run python3 compare_nodes.py --threshold 0.9 --k 20 --top-n 20
    uv run python3 compare_nodes.py --no-betweenness     # skip the slow one
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics.pairwise import cosine_similarity

EMBEDDINGS_FILE    = "specter2_embeddings.npy"
EMBEDDINGS_FT_FILE = "specter2finetuned_embeddings.npy"
INDEX_FILE         = "specter2_index.csv"

CENTRALITY_LABELS = {
    "strength":    "Weighted degree (strength)",
    "pagerank":    "Weighted PageRank",
    "eigenvector": "Eigenvector centrality",
    "betweenness": "Weighted betweenness",
}


# ---------------------------------------------------------------- graph build

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


# ---------------------------------------------------------------- centralities

def compute_centralities(G: nx.Graph, n: int, include_betweenness: bool) -> dict:
    out = {}

    # weighted degree (strength)
    out["strength"] = np.array(
        [s for _, s in sorted(G.degree(weight="weight"), key=lambda kv: kv[0])],
        dtype=float,
    )

    # weighted PageRank — robust on disconnected graphs
    pr = nx.pagerank(G, weight="weight") if G.number_of_edges() else {i: 1 / n for i in range(n)}
    out["pagerank"] = np.array([pr.get(i, 0.0) for i in range(n)], dtype=float)

    # eigenvector centrality on the LCC; zero elsewhere
    ev = np.zeros(n, dtype=float)
    if G.number_of_edges():
        lcc_nodes = max(nx.connected_components(G), key=len)
        if len(lcc_nodes) >= 3:
            H = G.subgraph(lcc_nodes)
            try:
                ev_dict = nx.eigenvector_centrality_numpy(H, weight="weight")
                for node, val in ev_dict.items():
                    ev[node] = float(val)
            except Exception:
                pass
    out["eigenvector"] = ev

    # weighted betweenness, distance = 1 - weight
    if include_betweenness and G.number_of_edges():
        H = G.copy()
        for u, v, d in H.edges(data=True):
            d["distance"] = max(1e-6, 1.0 - d["weight"])
        bc = nx.betweenness_centrality(H, weight="distance", normalized=True)
        out["betweenness"] = np.array([bc.get(i, 0.0) for i in range(n)], dtype=float)
    elif include_betweenness:
        out["betweenness"] = np.zeros(n, dtype=float)

    return out


# ---------------------------------------------------------------- kNN Jaccard

def topk_neighbours(embeddings: np.ndarray, k: int) -> np.ndarray:
    """For each row return indices of the top-k most similar OTHER rows."""
    sim = cosine_similarity(embeddings).astype(np.float32)
    np.fill_diagonal(sim, -np.inf)
    idx = np.argpartition(-sim, k, axis=1)[:, :k]
    return idx


def jaccard_per_node(nn_a: np.ndarray, nn_b: np.ndarray) -> np.ndarray:
    n, k = nn_a.shape
    out = np.empty(n, dtype=float)
    for i in range(n):
        a = set(nn_a[i].tolist())
        b = set(nn_b[i].tolist())
        u = len(a | b)
        out[i] = (len(a & b) / u) if u else 0.0
    return out


# ---------------------------------------------------------------- LaTeX helpers

def latex_escape(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    repl = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
            "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
            "^": r"\textasciicircum{}"}
    for k, v in repl.items():
        s = s.replace(k, v)
    return s


def truncate(s: str, n: int) -> str:
    s = str(s) if s is not None else ""
    return (s[: n - 1] + "…") if len(s) > n else s


# ---------------------------------------------------------------- per-paper CSV

def build_node_table(index: pd.DataFrame,
                     cent_o: dict, cent_f: dict,
                     knn_jaccard_primary: np.ndarray, k_primary: int) -> pd.DataFrame:
    df = index[["title", "authors", "year"]].copy()
    df.insert(0, "row_index", np.arange(len(df)))
    for key in cent_o:
        df[f"{key}_orig"] = cent_o[key]
        df[f"{key}_ft"]   = cent_f[key]
        df[f"{key}_delta"] = cent_f[key] - cent_o[key]
        df[f"{key}_rank_orig"] = pd.Series(cent_o[key]).rank(ascending=False, method="average").to_numpy()
        df[f"{key}_rank_ft"]   = pd.Series(cent_f[key]).rank(ascending=False, method="average").to_numpy()
        df[f"{key}_rank_shift"] = df[f"{key}_rank_ft"] - df[f"{key}_rank_orig"]
    df[f"knn{k_primary}_jaccard"] = knn_jaccard_primary
    return df


# ---------------------------------------------------------------- plots

def plot_scatters(cent_o: dict, cent_f: dict, spearman: dict, out_path: str):
    keys = list(cent_o.keys())
    rows = (len(keys) + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(11, 5 * rows))
    axes = axes.flatten()
    for i, key in enumerate(keys):
        ax = axes[i]
        x, y = cent_o[key], cent_f[key]
        # log scale only when all-positive and the dynamic range is wide
        use_log = (x.min() > 0 and y.min() > 0
                   and (x.max() / max(x.min(), 1e-12) > 100
                        or y.max() / max(y.min(), 1e-12) > 100))
        ax.scatter(x, y, s=8, alpha=0.45, color="#1f77b4")
        lo = min(x.min(), y.min())
        hi = max(x.max(), y.max())
        ax.plot([lo, hi], [lo, hi], "--", color="gray", alpha=0.6, linewidth=1)
        if use_log:
            ax.set_xscale("symlog")
            ax.set_yscale("symlog")
        ax.set_xlabel("original")
        ax.set_ylabel("fine-tuned")
        ax.set_title(f"{CENTRALITY_LABELS[key]}\nSpearman ρ = {spearman[key]:.3f}")
        ax.grid(True, alpha=0.3)
    for j in range(len(keys), len(axes)):
        axes[j].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_knn_sweep(ks: list, mean_orig_vs_ft: list, out_path: str):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ks, mean_orig_vs_ft, "-o", color="#d62728", linewidth=2)
    ax.set_xlabel("k (neighbourhood size)")
    ax.set_ylabel("mean Jaccard overlap (orig vs fine-tuned)")
    ax.set_title("Per-node kNN overlap between original and fine-tuned embeddings")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- LaTeX tables

def write_latex(path: str, threshold: float, k_primary: int,
                summary_rows: list, knn_rows: list,
                top_shift_per_metric: dict, knn_jaccard_primary: np.ndarray,
                node_df: pd.DataFrame, top_n: int):
    parts = []

    # --- Table 1: summary correlations + kNN stats
    parts.append(r"""\begin{table}[htbp]
\centering
\caption{Node-level comparison summary (graph threshold = """ + f"{threshold:.2f}" + r""", primary $k$ = """ + str(k_primary) + r""").}
\label{tab:node-summary}
\begin{tabular}{lrr}
\hline
Centrality measure & Spearman $\rho$ & Kendall $\tau$ \\
\hline
""")
    for metric, rho, tau in summary_rows:
        parts.append(f"{CENTRALITY_LABELS[metric]} & {rho:.4f} & {tau:.4f} \\\\\n")
    parts.append(r"""\hline
\end{tabular}

\bigskip

\begin{tabular}{lr}
\hline
\multicolumn{2}{l}{\textit{kNN Jaccard overlap (k = """ + str(k_primary) + r""")}} \\
\hline
""")
    parts.append(f"Mean   & {knn_jaccard_primary.mean():.4f} \\\\\n")
    parts.append(f"Std.\\ dev. & {knn_jaccard_primary.std():.4f} \\\\\n")
    parts.append(f"Median & {np.median(knn_jaccard_primary):.4f} \\\\\n")
    parts.append(f"Min    & {knn_jaccard_primary.min():.4f} \\\\\n")
    parts.append(f"Max    & {knn_jaccard_primary.max():.4f} \\\\\n")
    parts.append(r"""\hline
\end{tabular}
\end{table}
""")

    # --- Table 2: kNN sweep
    parts.append(r"""
\begin{table}[htbp]
\centering
\caption{Mean per-node Jaccard overlap of top-$k$ neighbours (original vs.\ fine-tuned).}
\label{tab:knn-sweep}
\begin{tabular}{rr}
\hline
$k$ & Mean Jaccard \\
\hline
""")
    for k_val, mean_j in knn_rows:
        parts.append(f"{k_val} & {mean_j:.4f} \\\\\n")
    parts.append(r"""\hline
\end{tabular}
\end{table}
""")

    # --- Tables 3..N: top shifted papers per centrality
    for metric, top_df in top_shift_per_metric.items():
        parts.append(r"""
\begin{table}[htbp]
\centering
\caption{Top """ + str(top_n) + r""" papers with the largest rank shift in """ + CENTRALITY_LABELS[metric].lower() + r""".}
\label{tab:shift-""" + metric + r"""}
\begin{tabular}{rlrrrr}
\hline
\# & Title (truncated) & Year & Rank orig & Rank ft & $\Delta$ rank \\
\hline
""")
        for i, row in enumerate(top_df.itertuples(index=False), 1):
            title = latex_escape(truncate(row.title, 70))
            year = latex_escape(row.year)
            r_o = row._asdict()[f"{metric}_rank_orig"]
            r_f = row._asdict()[f"{metric}_rank_ft"]
            d   = row._asdict()[f"{metric}_rank_shift"]
            parts.append(f"{i} & {title} & {year} & {r_o:.0f} & {r_f:.0f} & {d:+.0f} \\\\\n")
        parts.append(r"""\hline
\end{tabular}
\end{table}
""")

    # --- Table N+1: most-changed neighbourhoods (lowest kNN Jaccard)
    low = node_df.nsmallest(top_n, f"knn{k_primary}_jaccard")
    parts.append(r"""
\begin{table}[htbp]
\centering
\caption{Top """ + str(top_n) + r""" papers whose top-""" + str(k_primary) + r""" neighbourhood changed most (lowest Jaccard).}
\label{tab:knn-most-changed}
\begin{tabular}{rlrr}
\hline
\# & Title (truncated) & Year & Jaccard \\
\hline
""")
    for i, row in enumerate(low.itertuples(index=False), 1):
        title = latex_escape(truncate(row.title, 70))
        year = latex_escape(row.year)
        j = row._asdict()[f"knn{k_primary}_jaccard"]
        parts.append(f"{i} & {title} & {year} & {j:.3f} \\\\\n")
    parts.append(r"""\hline
\end{tabular}
\end{table}
""")

    Path(path).write_text("".join(parts))


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.94)
    ap.add_argument("--k", type=int, default=10, help="primary k for kNN comparison")
    ap.add_argument("--ks", type=int, nargs="+", default=[5, 10, 20, 50],
                    help="k values for the sweep plot")
    ap.add_argument("--top-n", type=int, default=15, help="top-N most-shifted papers per metric")
    ap.add_argument("--no-betweenness", action="store_true", help="skip weighted betweenness (slow)")
    ap.add_argument("--out-prefix", type=str, default="node_centrality")
    args = ap.parse_args()

    print("Loading embeddings & index…")
    emb_o = np.load(EMBEDDINGS_FILE).astype(np.float32)
    emb_f = np.load(EMBEDDINGS_FT_FILE).astype(np.float32)
    assert emb_o.shape[0] == emb_f.shape[0], "node count mismatch"
    n = emb_o.shape[0]
    index = pd.read_csv(INDEX_FILE).fillna("")

    print(f"Building graphs at threshold {args.threshold}…")
    G_o = build_graph(emb_o, args.threshold)
    G_f = build_graph(emb_f, args.threshold)
    print(f"  original:   {G_o.number_of_edges()} edges")
    print(f"  fine-tuned: {G_f.number_of_edges()} edges")

    include_bet = not args.no_betweenness
    print("Computing centralities for original…")
    cent_o = compute_centralities(G_o, n, include_bet)
    print("Computing centralities for fine-tuned…")
    cent_f = compute_centralities(G_f, n, include_bet)

    # primary kNN
    print(f"Computing kNN Jaccard at k={args.k}…")
    nn_o = topk_neighbours(emb_o, args.k)
    nn_f = topk_neighbours(emb_f, args.k)
    knn_jaccard = jaccard_per_node(nn_o, nn_f)

    # node table
    node_df = build_node_table(index, cent_o, cent_f, knn_jaccard, args.k)
    csv_path = f"{args.out_prefix}.csv"
    node_df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}")

    # summary stats
    spearman = {}
    kendall  = {}
    summary_rows = []
    for key in cent_o:
        rho, _ = spearmanr(cent_o[key], cent_f[key])
        tau, _ = kendalltau(cent_o[key], cent_f[key])
        spearman[key] = float(rho)
        kendall[key]  = float(tau)
        summary_rows.append((key, float(rho), float(tau)))

    # kNN sweep
    print("Sweeping k for kNN overlap…")
    knn_rows = []
    means = []
    for k_val in args.ks:
        nn_o_k = topk_neighbours(emb_o, k_val)
        nn_f_k = topk_neighbours(emb_f, k_val)
        j = jaccard_per_node(nn_o_k, nn_f_k)
        knn_rows.append((k_val, float(j.mean())))
        means.append(float(j.mean()))

    # text summary
    summary_path = f"{args.out_prefix}_summary.txt"
    lines = ["=== Centrality rank-correlation (orig vs fine-tuned) ==="]
    for key in cent_o:
        lines.append(f"  {CENTRALITY_LABELS[key]:32s}  Spearman ρ = {spearman[key]: .4f}   Kendall τ = {kendall[key]: .4f}")
    lines.append("")
    lines.append(f"=== kNN Jaccard (k={args.k}) ===")
    lines.append(f"  mean = {knn_jaccard.mean():.4f}   std = {knn_jaccard.std():.4f}   median = {np.median(knn_jaccard):.4f}")
    lines.append(f"  min  = {knn_jaccard.min():.4f}   max = {knn_jaccard.max():.4f}")
    lines.append("")
    lines.append("=== kNN Jaccard sweep ===")
    for k_val, m in knn_rows:
        lines.append(f"  k = {k_val:>3d}   mean Jaccard = {m:.4f}")
    Path(summary_path).write_text("\n".join(lines) + "\n")
    print(f"Wrote {summary_path}")
    print("\n" + "\n".join(lines))

    # plots
    scatter_path = f"{args.out_prefix}_scatter.png"
    plot_scatters(cent_o, cent_f, spearman, scatter_path)
    print(f"Wrote {scatter_path}")
    knn_plot_path = "knn_overlap_vs_k.png"
    plot_knn_sweep(args.ks, means, knn_plot_path)
    print(f"Wrote {knn_plot_path}")

    # top-shifted per metric (largest absolute rank change)
    top_shift = {}
    for key in cent_o:
        col = f"{key}_rank_shift"
        top_shift[key] = node_df.reindex(node_df[col].abs().sort_values(ascending=False).index).head(args.top_n)

    # LaTeX tables
    tex_path = f"{args.out_prefix}.tex"
    write_latex(tex_path, args.threshold, args.k, summary_rows, knn_rows,
                top_shift, knn_jaccard, node_df, args.top_n)
    print(f"Wrote {tex_path}")


if __name__ == "__main__":
    main()
