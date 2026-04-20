"""
Sweep the cosine-similarity threshold and plot, for the original and
fine-tuned SPECTER2 graphs:

  - Largest connected component (fraction of nodes) vs threshold
        →  lcc_vs_threshold.png

Usage:
    uv run python3 plot_lcc_vs_threshold.py
    uv run python3 plot_lcc_vs_threshold.py --min 0.6 --max 0.98 --steps 25
"""
import argparse

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

EMBEDDINGS_FILE    = "specter2_embeddings.npy"
EMBEDDINGS_FT_FILE = "specter2finetuned_embeddings.npy"


def precompute_sims(embeddings: np.ndarray):
    """Compute upper-triangle similarity vector once; reuse across thresholds."""
    sim = cosine_similarity(embeddings).astype(np.float32)
    np.fill_diagonal(sim, 0.0)
    iu, ju = np.triu_indices(len(embeddings), k=1)
    return iu, ju, sim[iu, ju]


def build_graph(iu, ju, sims, threshold, n_nodes):
    mask = sims >= threshold
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    G.add_weighted_edges_from(
        zip(iu[mask].tolist(), ju[mask].tolist(), sims[mask].astype(float).tolist())
    )
    return G


def lcc_size(G: nx.Graph) -> int:
    if G.number_of_nodes() == 0:
        return 0
    return max(len(c) for c in nx.connected_components(G))


def sweep(iu, ju, sims, thresholds, n, label):
    sizes = []
    for t in tqdm(thresholds, desc=f"sweeping {label}"):
        G = build_graph(iu, ju, sims, t, n)
        sizes.append(lcc_size(G))
    return np.array(sizes)


def plot_curve(thresholds, y_orig, y_ft, n, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, y_orig / n, "-o", label="original",   color="#1f77b4", markersize=4, linewidth=2)
    ax.plot(thresholds, y_ft   / n, "-o", label="fine-tuned", color="#d62728", markersize=4, linewidth=2)
    ax.set_xlabel("cosine-similarity threshold")
    ax.set_ylabel("LCC size (fraction of nodes)")
    ax.set_title(f"Largest connected component vs. threshold  (n = {n} papers)")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min", type=float, default=0.6)
    ap.add_argument("--max", type=float, default=0.98)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--out", type=str, default="lcc_vs_threshold.png")
    ap.add_argument("--csv", type=str, default="lcc_vs_threshold.csv")
    args = ap.parse_args()

    print("Loading embeddings…")
    emb_o = np.load(EMBEDDINGS_FILE).astype(np.float32)
    emb_f = np.load(EMBEDDINGS_FT_FILE).astype(np.float32)
    assert emb_o.shape[0] == emb_f.shape[0], "node count mismatch"
    n = emb_o.shape[0]

    print("Precomputing similarities…")
    iu_o, ju_o, s_o = precompute_sims(emb_o)
    iu_f, ju_f, s_f = precompute_sims(emb_f)

    thresholds = np.linspace(args.min, args.max, args.steps)
    lcc_o = sweep(iu_o, ju_o, s_o, thresholds, n, "original")
    lcc_f = sweep(iu_f, ju_f, s_f, thresholds, n, "fine-tuned")

    plot_curve(thresholds, lcc_o, lcc_f, n, args.out)

    import pandas as pd
    df = pd.DataFrame({
        "threshold":     thresholds,
        "lcc_orig":      lcc_o,
        "lcc_frac_orig": lcc_o / n,
        "lcc_ft":        lcc_f,
        "lcc_frac_ft":   lcc_f / n,
    })
    df.to_csv(args.csv, index=False)
    print(f"Wrote {args.csv}")
    print("\n" + df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
