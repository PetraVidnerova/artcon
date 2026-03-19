"""
Cluster papers using UMAP + HDBSCAN.

Feature sources:
  embeddings  — SPECTER2 embeddings (696×768), UMAP → HDBSCAN
  keywords    — TF-IDF on extracted keyphrases,  UMAP → HDBSCAN
  stance      — thesis stance vectors (5-dim, no UMAP needed), HDBSCAN directly

Output: ArtCon_clusters.csv with columns
  cluster_embeddings, cluster_keywords, cluster_stance  (−1 = noise)

Usage:
    uv run python3 cluster_papers.py               # all methods
    uv run python3 cluster_papers.py --method embeddings
    uv run python3 cluster_papers.py --method keywords
    uv run python3 cluster_papers.py --method stance
    uv run python3 cluster_papers.py --method louvain_embeddings
    uv run python3 cluster_papers.py --method louvain_keywords
    uv run python3 cluster_papers.py --method embeddings_ft
    uv run python3 cluster_papers.py --method louvain_embeddings_ft

Feature sources:
  louvain     — Louvain community detection on the document similarity graph
                (no noise label — all papers assigned to a community)
"""
import argparse
import numpy as np
import pandas as pd
import umap
import hdbscan
from collections import Counter

EMBEDDINGS_FILE    = "specter2_embeddings.npy"
EMBEDDINGS_FT_FILE = "specter2finetuned_embeddings.npy"
INDEX_FILE         = "specter2_index.csv"
KEYWORDS_FILE   = "ArtCon_keywords.csv"
THESES_FILE     = "ArtCon_theses.csv"
OUTPUT_FILE     = "ArtCon_clusters.csv"

UMAP_N_COMPONENTS = 15
UMAP_MIN_DIST     = 0.0

HDBSCAN_MIN_CLUSTER_SIZE = 30
HDBSCAN_MIN_SAMPLES      = 3

THESIS_KEYS = [
    "current_possibility", "future_possibility", "functionalism",
    "computational_functionalism", "biology",
]
STANCE_ENC = {"supports": 1, "opposes": -1, "neutral": 0}


def load_embeddings():
    print("Loading SPECTER2 embeddings…")
    mat = np.load(EMBEDDINGS_FILE)
    print(f"  {mat.shape[0]} papers × {mat.shape[1]} dims")
    return mat, 8   # n_neighbors tuned for embedding space


def load_embeddings_ft():
    print("Loading fine-tuned SPECTER2 embeddings…")
    mat = np.load(EMBEDDINGS_FT_FILE)
    print(f"  {mat.shape[0]} papers × {mat.shape[1]} dims")
    return mat, 8


def load_keyword_tfidf():
    from sklearn.feature_extraction.text import TfidfVectorizer
    print("Building TF-IDF matrix from keywords…")
    df = pd.read_csv(KEYWORDS_FILE)
    docs = [
        " ".join(
            k.strip().lower().replace(" ", "_")
            for k in str(row.get("keywords") or "").split(";") if k.strip()
        )
        for _, row in df.iterrows()
    ]
    vec = TfidfVectorizer(token_pattern=r"[^\s]+")
    mat = vec.fit_transform(docs).toarray().astype(np.float32)
    n_empty = sum(1 for d in docs if not d.strip())
    print(f"  {mat.shape[0]} papers × {mat.shape[1]} keyword features  ({n_empty} with no keywords)")
    return mat, 15  # n_neighbors tuned for keyword space


def load_stance():
    print("Building stance matrix from thesis classifications…")
    df = pd.read_csv(THESES_FILE)
    mat = np.array([
        [STANCE_ENC.get(str(df.iloc[i][k + "__stance"]), 0) for k in THESIS_KEYS]
        for i in range(len(df))
    ], dtype=np.float32)
    print(f"  {mat.shape[0]} papers × {mat.shape[1]} stance features")
    return mat   # no n_neighbors — will cluster directly without UMAP


def run_clustering(matrix, n_neighbors, name):
    """UMAP → HDBSCAN for high-dimensional feature spaces."""
    print(f"UMAP reduction for {name} (n_neighbors={n_neighbors})…")
    reducer = umap.UMAP(
        n_components=UMAP_N_COMPONENTS,
        n_neighbors=n_neighbors,
        min_dist=UMAP_MIN_DIST,
        metric="cosine",
        random_state=42,
    )
    reduced = reducer.fit_transform(matrix)
    return _hdbscan(reduced, name, min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE)


def _louvain_on_graph(G, n_nodes):
    """Run Louvain on a pre-built graph, collapse singletons to noise."""
    import networkx as nx
    from networkx.algorithms.community import louvain_communities
    communities = louvain_communities(G, weight="weight", seed=42)
    communities = sorted(communities, key=len, reverse=True)
    real    = [c for c in communities if len(c) > 1]
    n_noise = sum(len(c) for c in communities if len(c) == 1)
    labels  = [-1] * n_nodes
    for cid, community in enumerate(real):
        for node in community:
            labels[node] = cid
    print(f"  {len(real)} communities, {n_noise} singletons → noise")
    counts = Counter(labels)
    for cid in sorted(c for c in counts if c >= 0):
        print(f"    community {cid}: {counts[cid]} papers")
    return labels


def run_louvain_embeddings():
    """Louvain on cosine-similarity graph (SPECTER2 embeddings)."""
    import networkx as nx
    from sklearn.metrics.pairwise import cosine_similarity
    threshold = 0.94
    print("Louvain (embeddings): loading embeddings…")
    embeddings = np.load(EMBEDDINGS_FILE).astype(np.float32)
    sim_matrix = cosine_similarity(embeddings)
    print(f"  Building similarity graph (threshold={threshold})…")
    G = nx.Graph()
    G.add_nodes_from(range(len(embeddings)))
    i_idx, j_idx = np.triu_indices(len(embeddings), k=1)
    sims = sim_matrix[i_idx, j_idx]
    mask = sims >= threshold
    for i, j, w in zip(i_idx[mask], j_idx[mask], sims[mask]):
        G.add_edge(int(i), int(j), weight=float(w))
    print(f"  {G.number_of_edges()} edges")
    return _louvain_on_graph(G, len(embeddings))


def run_louvain_embeddings_ft():
    """Louvain on cosine-similarity graph (fine-tuned SPECTER2 embeddings)."""
    import networkx as nx
    from sklearn.metrics.pairwise import cosine_similarity
    threshold = 0.94
    print("Louvain (fine-tuned embeddings): loading embeddings…")
    embeddings = np.load(EMBEDDINGS_FT_FILE).astype(np.float32)
    sim_matrix = cosine_similarity(embeddings)
    print(f"  Building similarity graph (threshold={threshold})…")
    G = nx.Graph()
    G.add_nodes_from(range(len(embeddings)))
    i_idx, j_idx = np.triu_indices(len(embeddings), k=1)
    sims = sim_matrix[i_idx, j_idx]
    mask = sims >= threshold
    for i, j, w in zip(i_idx[mask], j_idx[mask], sims[mask]):
        G.add_edge(int(i), int(j), weight=float(w))
    print(f"  {G.number_of_edges()} edges")
    return _louvain_on_graph(G, len(embeddings))


def run_louvain_keywords():
    """Louvain on keyword co-occurrence graph."""
    import networkx as nx
    min_shared = 2
    print("Louvain (keywords): loading keywords…")
    df = pd.read_csv(KEYWORDS_FILE)
    kw_sets = [
        set(k.strip().lower() for k in str(row.get("keywords") or "").split(";") if k.strip())
        for _, row in df.iterrows()
    ]
    n = len(kw_sets)
    print(f"  Building keyword co-occurrence graph (min shared={min_shared})…")
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            shared = len(kw_sets[i] & kw_sets[j])
            if shared >= min_shared:
                G.add_edge(i, j, weight=shared)
    print(f"  {G.number_of_edges()} edges")
    return _louvain_on_graph(G, n)


def run_stance_clustering(matrix):
    """Direct HDBSCAN on 5-dim stance vectors (no UMAP needed)."""
    return _hdbscan(matrix, "stance", min_cluster_size=15)


def _hdbscan(matrix, name, min_cluster_size):
    print(f"HDBSCAN clustering for {name}…")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=2,
        metric="euclidean",
        cluster_selection_method="leaf",
    )
    labels = clusterer.fit_predict(matrix)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = (labels == -1).sum()
    print(f"  {n_clusters} clusters, {n_noise} noise points")
    counts = Counter(labels)
    for cid in sorted(counts):
        tag = f"cluster {cid}" if cid >= 0 else "noise"
        print(f"    {tag}: {counts[cid]} papers")
    return labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method",
                        choices=["embeddings", "keywords", "stance",
                                 "louvain_embeddings", "louvain_keywords",
                                 "embeddings_ft", "louvain_embeddings_ft", "all"],
                        default="all")
    args = parser.parse_args()

    # Start from existing clusters file if present so previous results are preserved
    import os
    if os.path.exists(OUTPUT_FILE) and args.method != "all":
        index = pd.read_csv(OUTPUT_FILE)
    else:
        index = pd.read_csv(INDEX_FILE)

    do_emb          = args.method in ("embeddings",           "all")
    do_kw           = args.method in ("keywords",             "all")
    do_stance       = args.method in ("stance",               "all")
    do_louv_emb     = args.method in ("louvain_embeddings",   "all")
    do_louv_kw      = args.method in ("louvain_keywords",     "all")
    do_emb_ft       = args.method in ("embeddings_ft",        "all")
    do_louv_emb_ft  = args.method in ("louvain_embeddings_ft","all")

    if do_emb:
        mat, nn = load_embeddings()
        index["cluster_embeddings"] = run_clustering(mat, nn, "embeddings")
    elif "cluster_embeddings" not in index.columns:
        index["cluster_embeddings"] = -1

    if do_kw:
        mat, nn = load_keyword_tfidf()
        index["cluster_keywords"] = run_clustering(mat, nn, "keywords")
    elif "cluster_keywords" not in index.columns:
        index["cluster_keywords"] = -1

    if do_stance:
        mat = load_stance()
        index["cluster_stance"] = run_stance_clustering(mat)
    elif "cluster_stance" not in index.columns:
        index["cluster_stance"] = -1

    if do_louv_emb:
        index["cluster_louvain_embeddings"] = run_louvain_embeddings()
    elif "cluster_louvain_embeddings" not in index.columns:
        index["cluster_louvain_embeddings"] = -1

    if do_louv_kw:
        index["cluster_louvain_keywords"] = run_louvain_keywords()
    elif "cluster_louvain_keywords" not in index.columns:
        index["cluster_louvain_keywords"] = -1

    if do_emb_ft and os.path.exists(EMBEDDINGS_FT_FILE):
        mat, nn = load_embeddings_ft()
        index["cluster_embeddings_ft"] = run_clustering(mat, nn, "embeddings_ft")
    elif "cluster_embeddings_ft" not in index.columns:
        index["cluster_embeddings_ft"] = -1

    if do_louv_emb_ft and os.path.exists(EMBEDDINGS_FT_FILE):
        index["cluster_louvain_embeddings_ft"] = run_louvain_embeddings_ft()
    elif "cluster_louvain_embeddings_ft" not in index.columns:
        index["cluster_louvain_embeddings_ft"] = -1

    index.to_csv(OUTPUT_FILE, index=False)
    print(f"\nWritten {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
