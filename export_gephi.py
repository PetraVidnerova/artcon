"""
Export ARTCON similarity graph to GEXF for Gephi.

Node attributes: title, authors, year, doi, openalex_id
Node viz:        position (UMAP), colour (by year), size
Edge weight:     cosine similarity

Usage:
    uv run python3 export_gephi.py              # default threshold 0.8
    uv run python3 export_gephi.py --threshold 0.75
"""
import argparse
import numpy as np
import pandas as pd
import networkx as nx
import umap
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

EMBEDDINGS_FILE = "specter2_embeddings.npy"
INDEX_FILE      = "specter2_index.csv"
OUTPUT_GEXF     = "artcon_graph.gexf"

YEAR_MIN, YEAR_MAX = 1987, 2025


def year_to_rgb(year_str: str) -> tuple[int, int, int]:
    try:
        y = int(year_str)
    except (ValueError, TypeError):
        return (136, 136, 136)
    t = max(0.0, min(1.0, (y - YEAR_MIN) / (YEAR_MAX - YEAR_MIN)))
    return (int(30 + t * 220), int(100 - t * 60), int(220 - t * 190))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.96,
                        help="Minimum cosine similarity for an edge to be included (default: 0.96)")
    args = parser.parse_args()

    print(f"Loading data…")
    embeddings = np.load(EMBEDDINGS_FILE).astype(np.float32)
    index = pd.read_csv(INDEX_FILE).fillna("")

    print("Computing UMAP layout…")
    reducer = umap.UMAP(n_components=2, metric="cosine", random_state=42, n_neighbors=15)
    coords = reducer.fit_transform(embeddings)
    # Scale to Gephi-friendly coordinate range
    scale = 1000
    coords = (coords - coords.min(axis=0)) / (coords.max(axis=0) - coords.min(axis=0)) * scale

    print("Computing cosine similarity…")
    sim_matrix = cosine_similarity(embeddings)

    print("Building graph…")
    G = nx.Graph()

    # --- Nodes ---
    for i, row in index.iterrows():
        r, g, b = year_to_rgb(row.get("year", ""))
        G.add_node(
            i,
            label=row.get("title", f"Entry {i}")[:80],
            title=row.get("title", ""),
            authors=row.get("authors", ""),
            year=str(row.get("year", "")),
            doi=row.get("doi", ""),
            openalex_id=row.get("openalex_id", ""),
            viz={
                "position": {"x": float(coords[i, 0]), "y": float(coords[i, 1]), "z": 0.0},
                "color":    {"r": r, "g": g, "b": b, "a": 1.0},
                "size":     10.0,
            },
        )

    # --- Edges ---
    i_idx, j_idx = np.triu_indices(len(embeddings), k=1)
    sims = sim_matrix[i_idx, j_idx]
    mask = sims >= args.threshold
    ei, ej, es = i_idx[mask], j_idx[mask], sims[mask]

    for k in tqdm(range(len(ei)), desc="Adding edges"):
        G.add_edge(int(ei[k]), int(ej[k]), weight=float(es[k]))

    print(f"\nGraph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges "
          f"(threshold={args.threshold})")

    print(f"Writing {OUTPUT_GEXF}…")
    nx.write_gexf(G, OUTPUT_GEXF)
    print(f"Done → {OUTPUT_GEXF}")
    print("\nIn Gephi:")
    print("  File → Open → artcon_graph.gexf")
    print("  Appearance → Edges → Weight  (to visualise edge strength)")
    print("  Filters → Edge Weight  (to dynamically adjust threshold)")


if __name__ == "__main__":
    main()
