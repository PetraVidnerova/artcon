"""
Sweep the cosine-similarity threshold and plot graph density vs threshold
for the original and fine-tuned SPECTER2 similarity graphs.

Usage:
    uv run python3 plot_density_vs_threshold.py
    uv run python3 plot_density_vs_threshold.py --min 0.5 --max 0.99 --steps 50
    uv run python3 plot_density_vs_threshold.py --out density_vs_threshold.png
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

EMBEDDINGS_FILE    = "specter2_embeddings.npy"
EMBEDDINGS_FT_FILE = "specter2finetuned_embeddings.npy"


def upper_tri_sims(embeddings: np.ndarray) -> np.ndarray:
    sim = cosine_similarity(embeddings).astype(np.float32)
    iu, ju = np.triu_indices(len(embeddings), k=1)
    return sim[iu, ju]


def densities(sims: np.ndarray, thresholds: np.ndarray, n_nodes: int) -> np.ndarray:
    total_pairs = n_nodes * (n_nodes - 1) / 2  # == len(sims)
    sorted_sims = np.sort(sims)
    # count of sims >= t  ==  len - searchsorted(left)
    counts = len(sorted_sims) - np.searchsorted(sorted_sims, thresholds, side="left")
    return counts / total_pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min", type=float, default=0.5)
    ap.add_argument("--max", type=float, default=0.99)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--out", type=str, default="density_vs_threshold.png")
    ap.add_argument("--logy", action="store_true", help="use log scale for density axis")
    args = ap.parse_args()

    print("Loading embeddings…")
    emb_orig = np.load(EMBEDDINGS_FILE).astype(np.float32)
    emb_ft   = np.load(EMBEDDINGS_FT_FILE).astype(np.float32)
    assert emb_orig.shape[0] == emb_ft.shape[0], "node count mismatch"
    n = emb_orig.shape[0]

    print("Computing similarities…")
    sims_orig = upper_tri_sims(emb_orig)
    sims_ft   = upper_tri_sims(emb_ft)

    thresholds = np.linspace(args.min, args.max, args.steps)
    d_orig = densities(sims_orig, thresholds, n)
    d_ft   = densities(sims_ft,   thresholds, n)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, d_orig, label="original",   linewidth=2, color="#1f77b4")
    ax.plot(thresholds, d_ft,   label="fine-tuned", linewidth=2, color="#d62728")
    ax.set_xlabel("cosine-similarity threshold")
    ax.set_ylabel("graph density")
    ax.set_title(f"Density vs. threshold  (n = {n} papers)")
    if args.logy:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Wrote {args.out}")

    print(f"\n{'threshold':>10}  {'density_orig':>14}  {'density_ft':>12}")
    for t, a, b in zip(thresholds, d_orig, d_ft):
        print(f"{t:10.3f}  {a:14.6f}  {b:12.6f}")


if __name__ == "__main__":
    main()
