"""
Sweep the cosine-similarity threshold and plot, for the original and
fine-tuned SPECTER2 graphs:

  - Louvain modularity     vs threshold  →  modularity_vs_threshold.png

Usage:
    uv run python3 plot_modularity_silhouette_vs_threshold.py
    uv run python3 plot_modularity_silhouette_vs_threshold.py --min 0.6 --max 0.98 --steps 25
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms.community import louvain_communities, modularity as nx_modularity
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


def louvain_modularity(G: nx.Graph, resolution: float, seed: int) -> float:
    if G.number_of_edges() == 0:
        return float("nan")
    comms = louvain_communities(G, weight="weight", resolution=resolution, seed=seed)
    return float(nx_modularity(G, comms, weight="weight", resolution=resolution))


def sweep(iu, ju, sims, thresholds, n, resolution, seed, label):
    mods = []
    for t in tqdm(thresholds, desc=f"sweeping {label}"):
        G = build_graph(iu, ju, sims, t, n)
        mods.append(louvain_modularity(G, resolution, seed))
    return np.array(mods)


def plot_curve(thresholds, y_orig, y_ft, ylabel, title, out_path, ylim=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, y_orig, "-o", label="original",   color="#1f77b4", markersize=4, linewidth=2)
    ax.plot(thresholds, y_ft,   "-o", label="fine-tuned", color="#d62728", markersize=4, linewidth=2)
    ax.set_xlabel("cosine-similarity threshold", fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_title(title, fontsize=16)
    ax.tick_params(axis="both", labelsize=14)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min", type=float, default=0.6)
    ap.add_argument("--max", type=float, default=0.98)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--resolution", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mod-out", type=str, default="modularity_vs_threshold.png")
    ap.add_argument("--csv",     type=str, default="modularity_vs_threshold.csv")
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
    mod_o = sweep(iu_o, ju_o, s_o, thresholds, n, args.resolution, args.seed, "original")
    mod_f = sweep(iu_f, ju_f, s_f, thresholds, n, args.resolution, args.seed, "fine-tuned")

    plot_curve(thresholds, mod_o, mod_f,
               "Louvain modularity (weighted)",
               f"Modularity vs. threshold  (n = {n} papers)",
               args.mod_out)

    import pandas as pd
    df = pd.DataFrame({
        "threshold":       thresholds,
        "modularity_orig": mod_o,
        "modularity_ft":   mod_f,
    })
    df.to_csv(args.csv, index=False)
    print(f"Wrote {args.csv}")
    print("\n" + df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
