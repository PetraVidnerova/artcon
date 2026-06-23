# 1. Re-cluster with fine-tuned embeddings
uv run python3 cluster_papers.py --method embeddings_ft
uv run python3 cluster_papers.py --method louvain_embeddings_ft

# 2. Re-run BERTopic with fine-tuned embeddings
uv run python3 topic_model.py --ft

# 3. Build TF-IDF (lexical) graph and cluster it
uv run python3 embed_tfidf.py
uv run python3 cluster_papers.py --method tfidf
uv run python3 cluster_papers.py --method louvain_tfidf
uv run python3 topic_model.py --tfidf

# 4. Regenerate paper pages
uv run python3 export_paper_pages.py

# 5. Regenerate the graph HTML and the multi-graph comparison
uv run python3 export_html.py
uv run python3 compare_three_graphs.py