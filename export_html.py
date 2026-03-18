"""
Generate a standalone HTML graph using pyvis (vis.js).
Embed in any webpage with:
    <iframe src="graph.html" width="100%" height="800px" frameborder="0"></iframe>

Usage:
    uv run python3 export_html.py
    uv run python3 export_html.py --threshold 0.75 --output my_graph.html
"""
import argparse
import colorsys
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
KEYWORDS_FILE   = "ArtCon_keywords.csv"
CLUSTERS_FILE   = "ArtCon_clusters.csv"
TOPIC_LABELS_FILE = "topic_labels.json"
YEAR_MIN, YEAR_MAX = 1987, 2025
MIN_SHARED_KW      = 2     # default selected value for keyword edges
SIM_PRECOMPUTE_MIN = 0.92  # lower bound for pre-computing sim edges
SIM_STEPS = [0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98]
KW_STEPS  = [1, 2, 3]


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


def cluster_to_color(cid: int) -> str:
    """Map a cluster ID to a hex colour. Noise (-1) → gray."""
    if cid < 0:
        return "#555566"
    # Golden-angle hue stepping gives maximally distinct colours
    hue = (cid * 137.508) % 360
    r, g, b = colorsys.hls_to_rgb(hue / 360, 0.55, 0.70)
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"


def load_cluster_data():
    """Return dict of {method: (node_clusters, palette)} or {}."""
    if not os.path.exists(CLUSTERS_FILE):
        print("  No ArtCon_clusters.csv found — cluster colouring will be disabled.")
        return {}
    df = pd.read_csv(CLUSTERS_FILE)
    # Load BERTopic labels if available
    topic_labels = {}
    if os.path.exists(TOPIC_LABELS_FILE):
        with open(TOPIC_LABELS_FILE, encoding="utf-8") as f:
            topic_labels = json.load(f)

    result = {}
    for col, label in [("cluster_embeddings", "Clusters (embeddings)"),
                        ("cluster_keywords",   "Clusters (keywords)"),
                        ("cluster_stance",     "Clusters (stance)"),
                        ("cluster_louvain_embeddings", "Clusters (Louvain, embeddings)"),
                        ("cluster_louvain_keywords",   "Clusters (Louvain, keywords)"),
                        ("cluster_topic",      "Topics (BERTopic)")]:
        if col not in df.columns:
            continue
        node_clusters = {i: int(c) for i, c in enumerate(df[col])}
        unique_ids    = sorted(set(c for c in node_clusters.values() if c >= 0))
        palette       = {cid: cluster_to_color(cid) for cid in unique_ids}
        palette[-1]   = cluster_to_color(-1)
        # For topic clustering, use keyword labels in the legend
        if col == "cluster_topic" and topic_labels:
            leg_labels = {str(cid): topic_labels.get(str(cid), f"Topic {cid}") for cid in unique_ids}
            leg_labels["-1"] = topic_labels.get("-1", "Noise")
        else:
            leg_labels = {}
        result[col] = {"label": label, "nodes": node_clusters,
                       "palette": palette, "leg_labels": leg_labels}
        print(f"  {label}: {len(unique_ids)} clusters")
    return result


def build_similarity_edges(embeddings, sim_min):
    """Return compact list of [from, to, sim_value] for all pairs >= sim_min.
    Edge visual properties are computed client-side to keep the file small."""
    print(f"Computing cosine similarity edges (min={sim_min})…")
    sim_matrix = cosine_similarity(embeddings)
    i_idx, j_idx = np.triu_indices(len(embeddings), k=1)
    sims = sim_matrix[i_idx, j_idx]
    mask = sims >= sim_min
    ei, ej, es = i_idx[mask], j_idx[mask], sims[mask]
    edges = [[int(ei[k]), int(ej[k]), round(float(es[k]), 4)] for k in range(len(ei))]
    print(f"  {len(edges)} similarity edges pre-computed (min={sim_min})")
    return edges


def build_keyword_edges(n_nodes):
    """Return compact list of [from, to, shared_count] for all pairs with shared>=1.
    Edge visual properties are computed client-side to keep the file small."""
    print("Computing keyword overlap edges…")
    if not os.path.exists(KEYWORDS_FILE):
        print("  No keywords file found — skipping.")
        return [], 0
    with open(KEYWORDS_FILE, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    kw_sets = [
        set(k.strip().lower() for k in (row.get("keywords") or "").split(";") if k.strip())
        for row in rows
    ]
    edges = []
    for i in range(len(kw_sets)):
        for j in range(i + 1, len(kw_sets)):
            w = len(kw_sets[i] & kw_sets[j])
            if w >= 1:
                edges.append([i, j, w])
    max_shared = max(e[2] for e in edges) if edges else 1
    print(f"  {len(edges)} keyword edges pre-computed (min shared=1), max shared={max_shared}")
    return edges, max_shared


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.96,
                        help="Minimum cosine similarity for similarity edges (default: 0.96)")
    parser.add_argument("--output", default="graph.html",
                        help="Output HTML file (default: graph.html)")
    args = parser.parse_args()

    print("Loading data…")
    embeddings = np.load(EMBEDDINGS_FILE).astype(np.float32)
    index = pd.read_csv(INDEX_FILE).fillna("")
    theses_list, thesis_node_data = load_thesis_data()
    cluster_data = load_cluster_data()

    sim_edges            = build_similarity_edges(embeddings, SIM_PRECOMPUTE_MIN)
    kw_edges, max_shared = build_keyword_edges(len(index))

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

    # Add similarity edges at default threshold for initial view
    sim_range = max(1.0 - SIM_PRECOMPUTE_MIN, 1e-6)
    for e in tqdm([e for e in sim_edges if e[2] >= args.threshold], desc="Adding edges"):
        nd = (e[2] - SIM_PRECOMPUTE_MIN) / sim_range
        length = float(max(1, (1.0 - nd) ** 2 * 400))
        net.add_edge(e[0], e[1], length=length, width=4,
                     color={"color": "rgba(180,180,255,0.25)", "highlight": "rgba(255,200,50,0.9)"})

    print(f"Graph: {len(index)} nodes, {len(sim_edges)} sim edges, {len(kw_edges)} keyword edges")

    html = net.generate_html()

    # Expose vis.js DataSets and network as globals
    html = html.replace(
        "nodes = window.visNodes = new vis.DataSet(",
        "nodes = window.visNodes = new vis.DataSet(",
    )
    html = html.replace(
        "nodes = new vis.DataSet(",
        "nodes = window.visNodes = new vis.DataSet(",
    )
    html = html.replace(
        "edges = new vis.DataSet(",
        "edges = window.visEdges = new vis.DataSet(",
    )
    html = html.replace(
        "network = new vis.Network(",
        "network = window.visNetwork = new vis.Network(",
    )

    # Build thesis node colours lookup: node_id → {thesis_key → label}
    thesis_colors_js = json.dumps(thesis_node_data, ensure_ascii=False)
    theses_meta_js   = json.dumps(theses_list, ensure_ascii=False)

    # Build original colours lookup from index
    orig_colors = {i: year_to_color(str(row.get("year", "")))
                   for i, row in index.iterrows()}
    orig_colors_js = json.dumps(orig_colors)

    # Cluster data for JS: {method_key: {nodes: {id: cid}, palette: {cid: color}, label: str}}
    cluster_js = json.dumps({
        key: {
            "label":      val["label"],
            "nodes":      {str(k): v for k, v in val["nodes"].items()},
            "palette":    {str(k): v for k, v in val["palette"].items()},
            "leg_labels": val.get("leg_labels", {}),
        }
        for key, val in cluster_data.items()
    })

    # Compact edge arrays for JS
    sim_edges_js   = json.dumps(sim_edges)
    kw_edges_js    = json.dumps(kw_edges)
    sim_steps_js   = json.dumps(SIM_STEPS)
    kw_steps_js    = json.dumps(KW_STEPS)
    max_shared_js  = json.dumps(max_shared)

    injection = f"""
    <style>
      /* Graph mode toggle */
      #mode-toggle {{
        position: fixed;
        top: 12px; right: 12px;
        z-index: 999;
        display: flex;
        background: rgba(13,15,23,0.92);
        border: 1px solid #2a2a3a;
        border-radius: 8px;
        overflow: hidden;
        font-family: "Segoe UI", system-ui, sans-serif;
        font-size: 0.82rem;
        backdrop-filter: blur(4px);
      }}
      .mode-btn {{
        padding: 7px 16px;
        cursor: pointer;
        border: none;
        background: transparent;
        color: #7070a0;
        transition: background 0.15s, color 0.15s;
      }}
      .mode-btn.active {{
        background: rgba(120,120,200,0.35);
        color: #ffffff;
      }}
      .mode-btn:hover:not(.active) {{ background: rgba(80,80,120,0.3); color: #c0c0e0; }}

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

      /* Left overlay */
      #left-overlay {{
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
        max-width: 340px;
        backdrop-filter: blur(4px);
      }}
      #left-overlay label {{
        display: block;
        margin-bottom: 6px;
        font-size: 0.75rem;
        color: #7070a0;
        text-transform: uppercase;
        letter-spacing: 0.06em;
      }}
      #left-overlay select {{
        width: 100%;
        background: #1a1c2a;
        border: 1px solid #3a3a5a;
        color: #d0d0e8;
        border-radius: 5px;
        padding: 5px 8px;
        font-size: 0.82rem;
        cursor: pointer;
        outline: none;
        margin-bottom: 6px;
      }}
      #left-overlay select:focus {{ border-color: #6060aa; }}
      #thesis-legend, #cluster-legend {{
        display: none;
        margin-top: 8px;
        font-size: 0.78rem;
        line-height: 1.8;
      }}
      #cluster-legend {{
        max-height: 220px;
        overflow-y: auto;
      }}

      /* Threshold buttons */
      #threshold-section {{
        margin-top: 8px;
        padding-top: 8px;
        border-top: 1px solid #2a2a3a;
      }}
      #threshold-section label {{
        margin-bottom: 6px;
      }}
      .thr-buttons {{
        display: flex;
        flex-wrap: wrap;
        gap: 4px;
      }}
      .thr-btn {{
        padding: 3px 8px;
        font-size: 0.75rem;
        cursor: pointer;
        border: 1px solid #3a3a5a;
        border-radius: 4px;
        background: #1a1c2a;
        color: #7070a0;
        transition: background 0.12s, color 0.12s;
      }}
      .thr-btn.active {{
        background: rgba(120,120,200,0.35);
        border-color: #6060aa;
        color: #ffffff;
      }}
      .thr-btn:hover:not(.active) {{ background: rgba(80,80,120,0.3); color: #c0c0e0; }}
      .edge-count {{
        font-size: 0.72rem;
        color: #6060a0;
        margin-top: 4px;
      }}
      .leg-dot {{
        display: inline-block;
        width: 10px; height: 10px;
        border-radius: 50%;
        margin-right: 5px;
        vertical-align: middle;
      }}
    </style>

    <div id="mode-toggle">
      <button class="mode-btn active" id="btn-sim" onclick="switchMode('sim')">Similarity</button>
      <button class="mode-btn"        id="btn-kw"  onclick="switchMode('kw')">Keywords</button>
    </div>

    <div id="left-overlay">
      <label>Colour nodes by</label>
      <select id="color-select">
        <option value="">— year (default) —</option>
      </select>

      <label style="margin-top:6px">Highlight by thesis</label>
      <select id="thesis-select">
        <option value="">— none —</option>
      </select>

      <div id="thesis-legend">
        <div><span class="leg-dot" style="background:#44dd88"></span><span id="leg-supports">Supports</span></div>
        <div><span class="leg-dot" style="background:#ff4455"></span><span id="leg-opposes">Opposes</span></div>
        <div><span class="leg-dot" style="background:#333348"></span>Neutral / no abstract</div>
      </div>
      <div id="cluster-legend"></div>

      <div id="threshold-section">
        <label id="threshold-label">Similarity threshold</label>
        <div class="thr-buttons" id="thr-buttons"></div>
        <div class="edge-count" id="edge-count"></div>
      </div>
    </div>

    <script>
      const THESIS_NODE_DATA  = {thesis_colors_js};
      const THESES_META       = {theses_meta_js};
      const ORIG_COLORS       = {orig_colors_js};
      const CLUSTER_DATA      = {cluster_js};  // method_key → {{label, nodes, palette}}

      // Populate colour-by dropdown with cluster options
      const colorSel = document.getElementById("color-select");
      Object.entries(CLUSTER_DATA).forEach(([key, val]) => {{
        const opt = document.createElement("option");
        opt.value = key;
        opt.textContent = val.label;
        colorSel.appendChild(opt);
      }});

      // Populate thesis dropdown
      const thesisSel = document.getElementById("thesis-select");
      THESES_META.forEach(p => {{
        const grp = document.createElement("optgroup");
        grp.label = p.pro;
        const opt = document.createElement("option");
        opt.value = p.key;
        opt.textContent = p.pro + " / " + p.con;
        grp.appendChild(opt);
        thesisSel.appendChild(grp);
      }});

      const STANCE_COLOR = {{
        "supports": {{ background: "#44dd88", border: "#22bb66" }},
        "opposes":  {{ background: "#ff4455", border: "#cc2233" }},
        "neutral":  {{ background: "#333348", border: "#444460" }},
      }};

      function updateThesisLegend(key) {{
        const pair = THESES_META.find(p => p.key === key);
        if (!pair) return;
        document.getElementById("leg-supports").textContent = pair.pro;
        document.getElementById("leg-opposes").textContent  = pair.con;
      }}

      // ── Colour-by selector ──
      let currentColorMode = "";   // "" = year, "__cluster__" = cluster

      document.getElementById("color-select").addEventListener("change", function() {{
        currentColorMode = this.value;
        const isCluster = currentColorMode && CLUSTER_DATA[currentColorMode];
        document.getElementById("cluster-legend").style.display = isCluster ? "block" : "none";
        if (isCluster) buildClusterLegend(currentColorMode);
        applyColors();
      }});

      function buildClusterLegend(methodKey) {{
        const el = document.getElementById("cluster-legend");
        el.innerHTML = "";
        if (!methodKey || !CLUSTER_DATA[methodKey]) return;
        const {{ palette, leg_labels }} = CLUSTER_DATA[methodKey];
        const ids = Object.keys(palette).map(Number).sort((a,b) => a - b);
        ids.forEach(cid => {{
          const color = palette[String(cid)];
          const sid   = String(cid);
          const label = leg_labels[sid]
            ? (cid < 0 ? leg_labels[sid] : `T${{cid}}: ${{leg_labels[sid]}}`)
            : (cid < 0 ? "Noise (unclustered)" : "Cluster " + cid);
          el.innerHTML +=
            `<div><span class="leg-dot" style="background:${{color}}"></span>${{label}}</div>`;
        }});
      }}

      function applyColors() {{
        if (!window.visNodes) return;
        const thesisKey = document.getElementById("thesis-select").value;
        const updates = [];
        window.visNodes.forEach(n => {{
          const sid = String(n.id);
          if (thesisKey) {{
            const info   = THESIS_NODE_DATA[sid]?.[thesisKey];
            const stance = info?.stance || "neutral";
            const color  = STANCE_COLOR[stance] || STANCE_COLOR["neutral"];
            const size   = stance !== "neutral" ? 16 : 8;
            updates.push({{ id: n.id, color, size }});
          }} else if (currentColorMode && CLUSTER_DATA[currentColorMode]) {{
            const cd    = CLUSTER_DATA[currentColorMode];
            const cid   = String(cd.nodes[sid] ?? -1);
            const color = cd.palette[cid] || "#555566";
            updates.push({{ id: n.id, color, size: 12 }});
          }} else {{
            updates.push({{ id: n.id, color: ORIG_COLORS[sid] || "#888888", size: 12 }});
          }}
        }});
        window.visNodes.update(updates);
      }}

      // ── Graph mode switch + threshold buttons ──
      // Compact edge data: sim = [from, to, sim_value], kw = [from, to, shared_count]
      const SIM_EDGES    = {sim_edges_js};
      const KW_EDGES     = {kw_edges_js};
      const SIM_STEPS    = {sim_steps_js};
      const KW_STEPS     = {kw_steps_js};
      const MAX_SHARED   = {max_shared_js};
      const SIM_MIN      = {SIM_PRECOMPUTE_MIN};

      let currentMode   = "sim";
      let currentSimThr = {args.threshold};
      let currentKwThr  = {MIN_SHARED_KW};

      const thrButtons  = document.getElementById("thr-buttons");
      const thrLabel    = document.getElementById("threshold-label");
      const edgeCountEl = document.getElementById("edge-count");

      function makeSimEdge(e, i) {{
        const sim = e[2];
        const nd  = (sim - SIM_MIN) / (1.0 - SIM_MIN);
        return {{
          id: i, from: e[0], to: e[1],
          length: Math.max(1, Math.pow(1 - nd, 2) * 400),
          width: 4,
          color: {{color: "rgba(180,180,255,0.25)", highlight: "rgba(255,200,50,0.9)"}},
        }};
      }}

      function makeKwEdge(e, i) {{
        const w  = e[2];
        const nd = 1 - w / MAX_SHARED;
        return {{
          id: i, from: e[0], to: e[1],
          length: Math.max(1, nd * nd * 400),
          width: Math.max(1, Math.round(4 * w / MAX_SHARED + 1)),
          color: {{color: "rgba(180,255,180,0.25)", highlight: "rgba(255,200,50,0.9)"}},
          title: "shared keywords: " + w,
        }};
      }}

      function applyEdges() {{
        if (!window.visEdges || !window.visNetwork) return;
        let edges;
        if (currentMode === "sim") {{
          edges = SIM_EDGES.filter(e => e[2] >= currentSimThr).map(makeSimEdge);
        }} else {{
          edges = KW_EDGES.filter(e => e[2] >= currentKwThr).map(makeKwEdge);
        }}
        window.visEdges.clear();
        window.visEdges.add(edges);
        const bar = document.getElementById("loadingBar");
        if (bar) bar.style.display = "none";
        edgeCountEl.textContent = edges.length + " edges";
      }}

      function buildThrButtons() {{
        thrButtons.innerHTML = "";
        const steps  = currentMode === "sim" ? SIM_STEPS : KW_STEPS;
        const active = currentMode === "sim" ? currentSimThr : currentKwThr;
        steps.forEach(val => {{
          const btn = document.createElement("button");
          btn.className = "thr-btn" + (val === active ? " active" : "");
          btn.textContent = currentMode === "sim" ? val.toFixed(2) : val;
          btn.onclick = () => {{
            if (currentMode === "sim") currentSimThr = val;
            else currentKwThr = val;
            buildThrButtons();
            applyEdges();
          }};
          thrButtons.appendChild(btn);
        }});
        thrLabel.textContent = currentMode === "sim" ? "Similarity threshold" : "Min shared keywords";
        edgeCountEl.textContent = "";
      }}

      function switchMode(mode) {{
        if (mode === currentMode) return;
        currentMode = mode;
        document.getElementById("btn-sim").classList.toggle("active", mode === "sim");
        document.getElementById("btn-kw").classList.toggle("active", mode === "kw");
        buildThrButtons();
        applyEdges();
        window.visNetwork.fit({{ animation: {{ duration: 600, easingFunction: "easeInOutQuad" }} }});
      }}

      buildThrButtons();

      // ── Thesis selector ──
      thesisSel.addEventListener("change", function() {{
        const key = this.value;
        document.getElementById("thesis-legend").style.display = key ? "block" : "none";
        document.getElementById("cluster-legend").style.display = key ? "none" : (currentColorMode && CLUSTER_DATA[currentColorMode] ? "block" : "none");
        if (key) updateThesisLegend(key);
        applyColors();
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
