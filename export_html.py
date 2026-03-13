"""
Generate a standalone HTML graph using pyvis (vis.js).
Embed in any webpage with:
    <iframe src="graph.html" width="100%" height="800px" frameborder="0"></iframe>

Usage:
    uv run python3 export_html.py
    uv run python3 export_html.py --threshold 0.75 --output my_graph.html
"""
import argparse
import csv
import json
import os
import re
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from pyvis.network import Network
from tqdm import tqdm

EMBEDDINGS_FILE = "specter2_embeddings.npy"
INDEX_FILE      = "specter2_index.csv"
THESES_FILE     = "ArtCon_theses.csv"
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


def slugify(text: str) -> str:
    text = text.lower().rstrip(".")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text[:60]


def load_thesis_data():
    """Load thesis pair classifications. Returns (pairs_list, node_data_dict) or ([], {})."""
    if not os.path.exists(THESES_FILE):
        print("  No ArtCon_theses.csv found — thesis selector will be disabled.")
        return [], {}

    try:
        from theses import THESIS_PAIRS
    except ImportError:
        print("  No theses.py found — thesis selector will be disabled.")
        return [], {}

    with open(THESES_FILE, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    node_data = {}
    for i, row in enumerate(rows):
        node_data[i] = {}
        for pair in THESIS_PAIRS:
            key    = pair["key"]
            stance = row.get(f"{key}__stance", "")
            score  = row.get(f"{key}__score", "0") or "0"
            if stance:
                node_data[i][key] = {"stance": stance, "score": float(score)}

    theses_list = [
        {"key": p["key"], "pro": p["pro"], "con": p["con"]}
        for p in THESIS_PAIRS
    ]
    print(f"  Loaded thesis classifications for {len(rows)} nodes, {len(THESIS_PAIRS)} pairs.")
    return theses_list, node_data


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
    theses_list, thesis_node_data = load_thesis_data()

    print("Computing cosine similarity…")
    sim_matrix = cosine_similarity(embeddings)

    print("Building graph…")
    net = Network(
        height="98vh", width="100%",
        bgcolor="#0e1117", font_color="white",
        notebook=False, cdn_resources="in_line",
    )
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

    sim_range = max(es.max() - args.threshold, 1e-6)
    for k in tqdm(range(len(ei)), desc="Adding edges"):
        sim = float(es[k])
        normalised_dissim = (sim - args.threshold) / sim_range
        length = float(max(1, (1.0 - normalised_dissim) ** 2 * 400))
        net.add_edge(
            int(ei[k]), int(ej[k]),
            value=sim,
            length=length,
            width=4,
            color={"color": "rgba(180,180,255,0.25)", "highlight": "rgba(255,200,50,0.9)"},
        )

    print(f"Graph: {len(index)} nodes, {len(ei)} edges (threshold={args.threshold})")

    html = net.generate_html()

    # Expose vis.js nodes DataSet as a global so injected JS can update colours
    html = html.replace(
        "nodes = new vis.DataSet(",
        "nodes = window.visNodes = new vis.DataSet(",
    )

    # Build thesis node colours lookup: node_id → {thesis_key → label}
    thesis_colors_js = json.dumps(thesis_node_data, ensure_ascii=False)
    theses_meta_js   = json.dumps(theses_list, ensure_ascii=False)

    # Build original colours lookup from index
    orig_colors = {i: year_to_color(str(row.get("year", "")))
                   for i, row in index.iterrows()}
    orig_colors_js = json.dumps(orig_colors)

    injection = f"""
    <style>
      /* Navigation button styling */
      div.vis-navigation div.vis-button {{
        opacity: 0.5;
        filter: grayscale(1) brightness(1.8);
      }}
      div.vis-navigation div.vis-button:hover {{
        opacity: 1.0;
        filter: grayscale(1) brightness(2.2);
        box-shadow: 0 0 6px 2px rgba(255,255,255,0.4) !important;
      }}
      div.vis-navigation div.vis-button:active {{
        filter: grayscale(1) brightness(2.5);
        box-shadow: 0 0 3px 3px rgba(255,255,255,0.7) !important;
      }}

      /* Thesis selector overlay */
      #thesis-overlay {{
        position: fixed;
        top: 12px; left: 12px;
        z-index: 999;
        background: rgba(13,15,23,0.92);
        border: 1px solid #2a2a3a;
        border-radius: 8px;
        padding: 10px 14px;
        font-family: "Segoe UI", system-ui, sans-serif;
        font-size: 0.82rem;
        color: #c0c0d0;
        min-width: 260px;
        max-width: 320px;
        backdrop-filter: blur(4px);
      }}
      #thesis-overlay label {{
        display: block;
        margin-bottom: 6px;
        font-size: 0.75rem;
        color: #7070a0;
        text-transform: uppercase;
        letter-spacing: 0.06em;
      }}
      #thesis-select {{
        width: 100%;
        background: #1a1c2a;
        border: 1px solid #3a3a5a;
        color: #d0d0e8;
        border-radius: 5px;
        padding: 5px 8px;
        font-size: 0.82rem;
        cursor: pointer;
        outline: none;
      }}
      #thesis-select:focus {{ border-color: #6060aa; }}
      #thesis-legend {{
        display: none;
        margin-top: 10px;
        font-size: 0.78rem;
        line-height: 1.8;
      }}
      .leg-dot {{
        display: inline-block;
        width: 10px; height: 10px;
        border-radius: 50%;
        margin-right: 5px;
        vertical-align: middle;
      }}
    </style>

    <div id="thesis-overlay">
      <label>Highlight by thesis</label>
      <select id="thesis-select">
        <option value="">— show original colours —</option>
      </select>
      <div id="thesis-legend">
        <div><span class="leg-dot" style="background:#44dd88"></span><span id="leg-supports">Supports</span></div>
        <div><span class="leg-dot" style="background:#ff4455"></span><span id="leg-opposes">Opposes</span></div>
        <div><span class="leg-dot" style="background:#333348"></span>Neutral / no abstract</div>
      </div>
    </div>

    <script>
      const THESIS_NODE_DATA = {thesis_colors_js};
      const THESES_META      = {theses_meta_js};
      const ORIG_COLORS      = {orig_colors_js};

      // Populate dropdown with optgroups per pair
      const sel = document.getElementById("thesis-select");
      THESES_META.forEach(p => {{
        const grp = document.createElement("optgroup");
        grp.label = p.pro;
        const opt = document.createElement("option");
        opt.value = p.key;
        opt.textContent = p.pro + " / " + p.con;
        grp.appendChild(opt);
        sel.appendChild(grp);
      }});

      // Colour map
      const STANCE_COLOR = {{
        "supports": {{ background: "#44dd88", border: "#22bb66" }},
        "opposes":  {{ background: "#ff4455", border: "#cc2233" }},
        "neutral":  {{ background: "#333348", border: "#444460" }},
      }};

      // Update legend labels dynamically
      function updateLegend(key) {{
        const pair = THESES_META.find(p => p.key === key);
        if (!pair) return;
        document.getElementById("leg-supports").textContent = pair.pro;
        document.getElementById("leg-opposes").textContent  = pair.con;
      }}

      sel.addEventListener("change", function () {{
        const key = this.value;
        const legend = document.getElementById("thesis-legend");
        legend.style.display = key ? "block" : "none";
        if (key) updateLegend(key);

        if (!window.visNodes) return;
        const updates = [];
        window.visNodes.forEach(n => {{
          const sid = String(n.id);
          if (!key) {{
            updates.push({{ id: n.id, color: ORIG_COLORS[sid] || "#888888", size: 12 }});
          }} else {{
            const info = THESIS_NODE_DATA[sid]?.[key];
            const stance = info?.stance || "neutral";
            const color  = STANCE_COLOR[stance] || STANCE_COLOR["neutral"];
            const size   = stance !== "neutral" ? 16 : 8;
            updates.push({{ id: n.id, color, size }});
          }}
        }});
        window.visNodes.update(updates);
      }});
    </script>
    """

    html = html.replace("</body>", injection + "</body>")

    print(f"Writing {args.output}…")
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Done → {args.output}")
    print("\nEmbed in your webpage with:")
    print(f'  <iframe src="{args.output}" width="100%" height="800px" frameborder="0"></iframe>')


if __name__ == "__main__":
    main()
