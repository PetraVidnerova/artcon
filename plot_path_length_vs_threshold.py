"""
Sweep the cosine-similarity threshold and plot average shortest path length
vs threshold for the original and fine-tuned SPECTER2 similarity graphs.

Path length is computed on the largest connected component (LCC), using
edge distance = 1 - weight (so more-similar papers are "closer").

Because the LCC shrinks as the threshold rises, we also plot the LCC
fraction on a second axis so you can see when the graph falls apart.

Usage:
    uv run python3 plot_path_length_vs_threshold.py
    uv run python3 plot_path_length_vs_threshold.py --min 0.6 --max 0.95 --steps 20
    uv run python3 plot_path_length_vs_threshold.py --unweighted
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

EMBEDDINGS_FILE    = "specter2_embeddings.npy"
EMBEDDINGS_FT_FILE = "specter2finetuned_embeddings.npy"


def upper_tri(embeddings: np.ndarray):
    sim = cosine_similarity(embeddings).astype(np.float32)
    iu, ju = np.triu_indices(len(embeddings), k=1)
    return iu, ju, sim[iu, ju]


def build_graph(iu, ju, sims, threshold, n_nodes):
    mask = sims >= threshold
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    ei, ej, es = iu[mask], ju[mask], sims[mask]
    G.add_weighted_edges_from(zip(ei.tolist(), ej.tolist(), es.astype(float).tolist()))
    return G


def avg_path_length(G: nx.Graph, weighted: bool) -> tuple[float, float]:
    """Return (avg_shortest_path_on_LCC, LCC_fraction_of_nodes)."""
    n = G.number_of_nodes()
    if G.number_of_edges() == 0 or n == 0:
        return float("nan"), 0.0
    lcc_nodes = max(nx.connected_components(G), key=len)
    if len(lcc_nodes) < 2:
        return float("nan"), len(lcc_nodes) / n
    H = G.subgraph(lcc_nodes).copy()
    if weighted:
        for u, v, d in H.edges(data=True):
            d["distance"] = max(1e-6, 1.0 - d["weight"])
        apl = nx.average_shortest_path_length(H, weight="distance")
    else:
        apl = nx.average_shortest_path_length(H)
    return float(apl), len(lcc_nodes) / n


def sweep(iu, ju, sims, thresholds, n_nodes, weighted, label):
    apls, lccs = [], []
    for t in tqdm(thresholds, desc=f"sweeping {label}"):
        G = build_graph(iu, ju, sims, t, n_nodes)
        apl, lcc = avg_path_length(G, weighted)
        apls.append(apl)
        lccs.append(lcc)
    return np.array(apls), np.array(lccs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min", type=float, default=0.6)
    ap.add_argument("--max", type=float, default=0.95)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--unweighted", action="store_true",
                    help="use hop count instead of 1-weight as edge distance")
    ap.add_argument("--out", type=str, default="path_length_vs_threshold.png")
    args = ap.parse_args()

    weighted = not args.unweighted

    print("Loading embeddings…")
    emb_orig = np.load(EMBEDDINGS_FILE).astype(np.float32)
    emb_ft   = np.load(EMBEDDINGS_FT_FILE).astype(np.float32)
    assert emb_orig.shape[0] == emb_ft.shape[0], "node count mismatch"
    n = emb_orig.shape[0]

    print("Computing similarities…")
    iu_o, ju_o, s_o = upper_tri(emb_orig)
    iu_f, ju_f, s_f = upper_tri(emb_ft)

    thresholds = np.linspace(args.min, args.max, args.steps)
    apl_o, lcc_o = sweep(iu_o, ju_o, s_o, thresholds, n, weighted, "original")
    apl_f, lcc_f = sweep(iu_f, ju_f, s_f, thresholds, n, weighted, "fine-tuned")

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(thresholds, apl_o, "-o", label="original (APL)",   color="#1f77b4", markersize=4)
    ax1.plot(thresholds, apl_f, "-o", label="fine-tuned (APL)", color="#d62728", markersize=4)
    ax1.set_xlabel("cosine-similarity threshold")
    ylabel = "avg shortest path length  (weighted, dist = 1 - sim)" if weighted \
             else "avg shortest path length  (hops)"
    ax1.set_ylabel(ylabel)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(thresholds, lcc_o, "--", label="original (LCC frac)",   color="#1f77b4", alpha=0.5)
    ax2.plot(thresholds, lcc_f, "--", label="fine-tuned (LCC frac)", color="#d62728", alpha=0.5)
    ax2.set_ylabel("largest-component fraction")
    ax2.set_ylim(0, 1.02)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=9)

    ax1.set_title(f"Avg shortest path length vs. threshold  (n = {n} papers)")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Wrote {args.out}")

    print(f"\n{'threshold':>10}  {'apl_orig':>10}  {'lcc_orig':>9}  {'apl_ft':>10}  {'lcc_ft':>8}")
    for t, ao, lo, af, lf in zip(thresholds, apl_o, lcc_o, apl_f, lcc_f):
        print(f"{t:10.3f}  {ao:10.4f}  {lo:9.4f}  {af:10.4f}  {lf:8.4f}")


if __name__ == "__main__":
    main()
