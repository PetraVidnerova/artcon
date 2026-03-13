"""
Generate a standalone HTML graph using pyvis (vis.js).
Embed in any webpage with:
    <iframe src="graph.html" width="100%" height="800px" frameborder="0"></iframe>

Usage:
    uv run python3 export_html.py
    uv run python3 export_html.py --threshold 0.75 --output my_graph.html
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from pyvis.network import Network
from tqdm import tqdm

EMBEDDINGS_FILE = "specter2_embeddings.npy"
INDEX_FILE      = "specter2_index.csv"
YEAR_MIN, YEAR_MAX = 1987, 2025


def year_to_color(year_str: str) -> str:
    try:
        y = int(year_str)
    except (ValueError, TypeError):
        return "#888888"
    t = max(0.0, min(1.0, (y - YEAR_MIN) / (YEAR_MAX - YEAR_MIN)))
    r = int(30  + t * 220)
    g = int(100 - t * 60)
    b = int(220 - t * 190)
    return f"#{r:02x}{g:02x}{b:02x}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.96,
                        help="Minimum cosine similarity for an edge (default: 0.96)")
    parser.add_argument("--output", default="graph.html",
                        help="Output HTML file (default: graph.html)")
    args = parser.parse_args()

    print("Loading data…")
    embeddings = np.load(EMBEDDINGS_FILE).astype(np.float32)
    index = pd.read_csv(INDEX_FILE).fillna("")

    print("Computing cosine similarity…")
    sim_matrix = cosine_similarity(embeddings)

    print("Building graph…")
    net = Network(
        height="98vh", width="100%",
        bgcolor="#0e1117", font_color="white",
        notebook=False, cdn_resources="in_line",
    )
    # Physics: spring length encodes dissimilarity → similar nodes pulled close
    net.set_options("""
    {
      "nodes": { "font": { "size": 9, "color": "white" } },
      "edges": { "smooth": false },
      "interaction": {
        "hover": true,
        "tooltipDelay": 100,
        "zoomView": true,
        "zoomSpeed": 0.5,
        "navigationButtons": true
      },
      "physics": {
        "solver": "forceAtlas2Based",
        "forceAtlas2Based": {
          "springLength": 100,
          "springConstant": 0.08,
          "gravitationalConstant": -120,
          "centralGravity": 0.01,
          "damping": 0.9
        },
        "stabilization": { "iterations": 500 }
      }
    }
    """)

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
            size=12,
            color=year_to_color(row.get("year", "")),
        )

    i_idx, j_idx = np.triu_indices(len(embeddings), k=1)
    sims = sim_matrix[i_idx, j_idx]
    mask = sims >= args.threshold
    ei, ej, es = i_idx[mask], j_idx[mask], sims[mask]

    # Normalise edge length within actual similarity range [threshold, 1.0]
    sim_range = max(es.max() - args.threshold, 1e-6)
    for k in tqdm(range(len(ei)), desc="Adding edges"):
        sim = float(es[k])
        # Map: sim=threshold → length=400, sim=1.0 → length=1
        normalised_dissim = (sim - args.threshold) / sim_range  # 0=most similar, 1=least
        length = float(max(1, (1.0 - normalised_dissim) ** 2 * 400))
        net.add_edge(
            int(ei[k]), int(ej[k]),
            value=sim,
            length=length,
            width=4,
            color={"color": "rgba(180,180,255,0.25)", "highlight": "rgba(255,200,50,0.9)"},
        )

    print(f"Graph: {len(index)} nodes, {len(ei)} edges (threshold={args.threshold})")

    # Generate HTML and inject custom button styles
    html = net.generate_html()
    button_css = """
    <style>
      div.vis-navigation div.vis-button {
        opacity: 0.5;
        filter: grayscale(1) brightness(1.8);
      }
      div.vis-navigation div.vis-button:hover {
        opacity: 1.0;
        filter: grayscale(1) brightness(2.2);
        box-shadow: 0 0 6px 2px rgba(255, 255, 255, 0.4) !important;
      }
      div.vis-navigation div.vis-button:active {
        filter: grayscale(1) brightness(2.5);
        box-shadow: 0 0 3px 3px rgba(255, 255, 255, 0.7) !important;
      }
    </style>
    """
    html = html.replace("</head>", button_css + "</head>")

    print(f"Writing {args.output}…")
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Done → {args.output}")
    print("\nEmbed in your webpage with:")
    print(f'  <iframe src="{args.output}" width="100%" height="800px" frameborder="0"></iframe>')


if __name__ == "__main__":
    main()
