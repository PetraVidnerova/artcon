"""
Topic modelling with BERTopic using pre-computed SPECTER2 embeddings.

Pipeline:
    SPECTER2 embeddings (696×768)
    → UMAP (via BERTopic internals)
    → HDBSCAN (via BERTopic internals)
    → c-TF-IDF on abstracts → topic labels

Output:
    ArtCon_clusters.csv  — adds column  cluster_topic  (−1 = noise/no abstract)
    topic_labels.json    — {topic_id: "label string"} for display in graph

Usage:
    uv run python3 topic_model.py
"""
import json
import numpy as np
import pandas as pd
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer

EMBEDDINGS_FILE = "specter2_embeddings.npy"
THESES_FILE     = "ArtCon_theses.csv"
CLUSTERS_FILE   = "ArtCon_clusters.csv"
LABELS_FILE     = "topic_labels.json"

# UMAP settings (same as cluster_papers.py for consistency)
umap_model = UMAP(
    n_components=15,
    n_neighbors=8,
    min_dist=0.0,
    metric="cosine",
    random_state=42,
)

# HDBSCAN settings
hdbscan_model = HDBSCAN(
    min_cluster_size=15,
    min_samples=2,
    metric="euclidean",
    cluster_selection_method="leaf",
    prediction_data=True,   # required for BERTopic soft clustering
)


def main():
    print("Loading data…")
    embeddings = np.load(EMBEDDINGS_FILE)
    df = pd.read_csv(THESES_FILE).fillna("")
    docs = [str(r) if str(r).strip() else "" for r in df["abstract"]]
    print(f"  {len(docs)} documents, {sum(1 for d in docs if not d.strip())} without abstract")

    print("Fitting BERTopic…")
    vectorizer = CountVectorizer(
        stop_words="english",
        ngram_range=(1, 2),   # include bigrams for richer labels
        min_df=2,
    )
    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        calculate_probabilities=False,
        verbose=True,
        nr_topics="auto",
    )
    topics, _ = topic_model.fit_transform(docs, embeddings=embeddings)

    topic_info = topic_model.get_topic_info()
    print("\nTopics found:")
    for _, row in topic_info.iterrows():
        tid = row["Topic"]
        if tid == -1:
            print(f"  noise: {row['Count']} papers")
        else:
            words = ", ".join(w for w, _ in topic_model.get_topic(tid)[:6])
            print(f"  topic {tid}: {row['Count']} papers  [{words}]")

    # Build label map: topic_id → short label from top keywords
    labels = {}
    for _, row in topic_info.iterrows():
        tid = row["Topic"]
        if tid == -1:
            labels["-1"] = "Noise"
        else:
            words = [w for w, _ in topic_model.get_topic(tid)[:4]]
            labels[str(tid)] = ", ".join(words)

    with open(LABELS_FILE, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2, ensure_ascii=False)
    print(f"\nWritten {LABELS_FILE}")

    # Merge into clusters CSV
    if pd.io.common.file_exists(CLUSTERS_FILE):
        clusters_df = pd.read_csv(CLUSTERS_FILE)
    else:
        clusters_df = pd.read_csv("specter2_index.csv")

    clusters_df["cluster_topic"] = topics
    clusters_df.to_csv(CLUSTERS_FILE, index=False)
    print(f"Written {CLUSTERS_FILE}  (added cluster_topic column)")


if __name__ == "__main__":
    main()
