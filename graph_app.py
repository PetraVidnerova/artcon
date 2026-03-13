"""
Interactive document similarity graph using SPECTER2 embeddings.
Nodes = papers, edges = cosine similarity above threshold.
UMAP projects embeddings to 2D so similar papers appear close.
Uses pyvis (vis.js) embedded in Gradio.
"""
import numpy as np
import pandas as pd
import gradio as gr
import umap
from functools import lru_cache
from sklearn.metrics.pairwise import cosine_similarity
from pyvis.network import Network

EMBEDDINGS_FILE = "specter2_embeddings.npy"
INDEX_FILE      = "specter2_index.csv"


@lru_cache(maxsize=1)
def load_data():
    embeddings = np.load(EMBEDDINGS_FILE).astype(np.float32)
    index = pd.read_csv(INDEX_FILE).fillna("")
    return embeddings, index


@lru_cache(maxsize=1)
def compute_layout():
    embeddings, _ = load_data()
    reducer = umap.UMAP(n_components=2, metric="cosine", random_state=42, n_neighbors=15)
    coords = reducer.fit_transform(embeddings)
    # Normalise to pixel range
    scale = 2000
    coords = (coords - coords.min(axis=0)) / (coords.max(axis=0) - coords.min(axis=0)) * scale
    return coords


@lru_cache(maxsize=1)
def compute_similarity():
    embeddings, _ = load_data()
    return cosine_similarity(embeddings)


@lru_cache(maxsize=1)
def precompute_edges():
    sim_matrix = compute_similarity()
    n = sim_matrix.shape[0]
    i_idx, j_idx = np.triu_indices(n, k=1)
    sims = sim_matrix[i_idx, j_idx]
    return i_idx, j_idx, sims


def year_to_color(year_str: str) -> str:
    try:
        y = int(year_str)
    except (ValueError, TypeError):
        return "#888888"
    y_min, y_max = 1987, 2025
    t = max(0.0, min(1.0, (y - y_min) / (y_max - y_min)))
    r = int(30  + t * 220)
    g = int(100 - t * 60)
    b = int(220 - t * 190)
    return f"#{r:02x}{g:02x}{b:02x}"


def build_graph(threshold: float, physics: bool) -> tuple[str, str]:
    embeddings, index = load_data()
    coords = compute_layout()
    i_idx, j_idx, sims = precompute_edges()

    mask = sims >= threshold
    ei, ej, es = i_idx[mask], j_idx[mask], sims[mask]
    n_edges = int(mask.sum())

    net = Network(height="750px", width="100%", bgcolor="#0e1117", font_color="white",
                  notebook=False)
    net.toggle_physics(physics)

    for i, row in index.iterrows():
        tooltip = (
            f"<b>{row.get('title', '')}</b><br>"
            f"{row.get('authors', '')}<br>"
            f"Year: {row.get('year', '')}<br>"
            f"DOI: {row.get('doi', '')}"
        )
        net.add_node(
            i,
            label=str(row.get("year", "")),
            title=tooltip,
            x=float(coords[i, 0]),
            y=float(coords[i, 1]),
            size=8,
            color=year_to_color(row.get("year", "")),
            font={"size": 9, "color": "white"},
        )

    for k in range(len(ei)):
        w = float(es[k])
        net.add_edge(
            int(ei[k]), int(ej[k]),
            value=w,
            title=f"similarity: {w:.3f}",
            color={"color": "rgba(180,180,255,0.3)", "highlight": "rgba(255,200,50,0.9)"},
        )

    html = net.generate_html()
    info = f"**Nodes:** {len(index)}  |  **Edges shown:** {n_edges}"
    return html, info


# --- Gradio UI ---
with gr.Blocks(title="ARTCON Similarity Graph", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# ARTCON Document Similarity Graph")
    gr.Markdown("Nodes coloured by publication year (blue = older, red = recent). Hover nodes/edges for details.")

    with gr.Row():
        with gr.Column(scale=1):
            threshold = gr.Slider(
                minimum=0.0, maximum=1.0, value=0.96, step=0.01,
                label="Similarity threshold",
                info="Only show edges with cosine similarity ≥ this value",
            )
            physics = gr.Checkbox(label="Enable physics (force simulation)", value=False)
            run_btn = gr.Button("Update graph", variant="primary")
            info = gr.Markdown("")

        with gr.Column(scale=4):
            graph_html = gr.HTML()

    run_btn.click(
        fn=build_graph,
        inputs=[threshold, physics],
        outputs=[graph_html, info],
    )
    threshold.release(
        fn=build_graph,
        inputs=[threshold, physics],
        outputs=[graph_html, info],
    )
    demo.load(
        fn=build_graph,
        inputs=[threshold, physics],
        outputs=[graph_html, info],
    )

if __name__ == "__main__":
    demo.launch()
