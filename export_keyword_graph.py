"""
Build a keyword co-occurrence graph with a time slider.
Nodes  = keywords
Edges  = number of documents sharing both keywords (co-occurrence)
Slider = cumulative view up to selected year

Output: keyword_graph.html  (standalone, embeddable via <iframe>)
"""
import csv
import json
import re
import numpy as np
import networkx as nx
from collections import defaultdict, Counter

INPUT          = "ArtCon_keywords.csv"
OUTPUT         = "keyword_graph.html"
MIN_KW_FREQ    = 3   # keyword must appear in at least N documents
MIN_EDGE_WEIGHT = 2  # keyword pair must co-occur in at least N documents


def parse_keywords(kw_str: str) -> list[str]:
    if not kw_str or not kw_str.strip():
        return []
    return [k.strip().lower() for k in kw_str.split(";") if k.strip() and len(k.strip()) > 2]


def year_to_color(year: int, y_min: int, y_max: int) -> str:
    t = max(0.0, min(1.0, (year - y_min) / max(y_max - y_min, 1)))
    r = int(30  + t * 220)
    g = int(100 - t * 60)
    b = int(220 - t * 190)
    return f"#{r:02x}{g:02x}{b:02x}"


def main():
    print("Loading data…")
    with open(INPUT, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    docs = []
    for row in rows:
        try:
            year = int(row.get("year", "").strip())
        except ValueError:
            continue
        kws = parse_keywords(row.get("keywords", ""))
        if kws:
            docs.append({"year": year, "keywords": kws, "title": row.get("title", "")})

    print(f"  {len(docs)} documents with keywords and valid year")

    # Keyword frequencies (unique per document)
    kw_freq = Counter()
    for doc in docs:
        kw_freq.update(set(doc["keywords"]))

    valid_kws = {kw for kw, cnt in kw_freq.items() if cnt >= MIN_KW_FREQ}
    print(f"  {len(valid_kws)} keywords in >= {MIN_KW_FREQ} documents")

    for doc in docs:
        doc["keywords"] = [k for k in doc["keywords"] if k in valid_kws]
    docs = [d for d in docs if len(d["keywords"]) >= 2]

    # Co-occurrence: store list of years each pair co-occurs
    cooc_years = defaultdict(list)
    for doc in docs:
        kws = sorted(set(doc["keywords"]))
        for i in range(len(kws)):
            for j in range(i + 1, len(kws)):
                cooc_years[(kws[i], kws[j])].append(doc["year"])

    # Build full graph
    G = nx.Graph()
    for kw in valid_kws:
        G.add_node(kw, freq=kw_freq[kw])
    for (a, b), years in cooc_years.items():
        if len(years) >= MIN_EDGE_WEIGHT:
            G.add_edge(a, b, weight=len(years), years=sorted(years))
    G.remove_nodes_from(list(nx.isolates(G)))
    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Layout on full graph (fixed positions for stability during time animation)
    print("Computing layout…")
    pos = nx.spring_layout(G, weight="weight", seed=42, k=1.5, iterations=200)
    scale = 1200
    xs = np.array([pos[n][0] for n in G.nodes()])
    ys = np.array([pos[n][1] for n in G.nodes()])
    xs = (xs - xs.min()) / (xs.max() - xs.min() + 1e-9) * scale - scale / 2
    ys = (ys - ys.min()) / (ys.max() - ys.min() + 1e-9) * scale - scale / 2
    for i, node in enumerate(G.nodes()):
        G.nodes[node]["x"] = float(xs[i])
        G.nodes[node]["y"] = float(ys[i])

    # First year each keyword appeared
    kw_first_year = {}
    for doc in sorted(docs, key=lambda d: d["year"]):
        for kw in doc["keywords"]:
            if kw in G.nodes() and kw not in kw_first_year:
                kw_first_year[kw] = doc["year"]

    all_years = sorted(set(doc["year"] for doc in docs))
    y_min, y_max = all_years[0], all_years[-1]
    max_freq = max(kw_freq[n] for n in G.nodes())

    # Nodes JSON
    nodes_js = []
    for node in G.nodes():
        fy = kw_first_year.get(node, y_min)
        size = float(6 + 22 * np.log1p(kw_freq[node]) / np.log1p(max_freq))
        nodes_js.append({
            "id":         node,
            "label":      node,
            "title":      f"<b>{node}</b><br>Documents: {kw_freq[node]}<br>First seen: {fy}",
            "x":          float(G.nodes[node]["x"]),
            "y":          float(G.nodes[node]["y"]),
            "size":       size,
            "color":      year_to_color(fy, y_min, y_max),
            "first_year": fy,
            "hidden":     True,
            "fixed":      {"x": True, "y": True},
        })

    # Edges JSON
    edges_js = []
    max_weight = max(d["weight"] for _, _, d in G.edges(data=True))
    for i, (u, v, data) in enumerate(G.edges(data=True)):
        edges_js.append({
            "id":         f"e{i}",
            "from":       u,
            "to":         v,
            "years":      data["years"],
            "max_weight": data["weight"],
            "hidden":     True,
            "color":      {"color": "rgba(180,180,255,0.3)", "highlight": "rgba(255,200,50,0.9)"},
        })

    graph_data = json.dumps({
        "nodes":    nodes_js,
        "edges":    edges_js,
        "years":    all_years,
        "max_weight": max_weight,
    }, ensure_ascii=False)

    html = build_html(graph_data, y_min, y_max)

    with open(OUTPUT, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Done → {OUTPUT}")


def build_html(graph_data: str, y_min: int, y_max: int) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<link rel="stylesheet" href="https://unpkg.com/vis-network@9.1.9/dist/vis-network.min.css">
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ background: #0e1117; color: #e0e0e0; font-family: "Segoe UI", system-ui, sans-serif; overflow: hidden; }}

  #controls {{
    display: flex; align-items: center; gap: 1rem;
    padding: 0.6rem 1.2rem;
    background: #13151f; border-bottom: 1px solid #2a2a3a;
    height: 52px;
  }}
  #controls label {{ font-size: 0.85rem; color: #9090a0; white-space: nowrap; }}
  #year-slider {{
    flex: 1; accent-color: #7a7aff;
    cursor: pointer;
  }}
  #year-display {{
    font-size: 1.1rem; font-weight: 600; color: #ffffff;
    min-width: 3rem; text-align: center;
  }}
  .ctrl-btn {{
    background: rgba(80,80,120,0.3); border: 1px solid rgba(150,150,255,0.3);
    color: #c0c0ff; border-radius: 6px; padding: 0.3rem 0.7rem;
    cursor: pointer; font-size: 1rem; line-height: 1; white-space: nowrap;
  }}
  .ctrl-btn:hover {{ background: rgba(120,120,200,0.5); color: #fff; }}
  #play-btn {{
    background: rgba(120,120,200,0.2); border: 1px solid rgba(150,150,255,0.4);
    color: #b0b0ff; border-radius: 6px; padding: 0.3rem 0.9rem;
    cursor: pointer; font-size: 0.85rem; white-space: nowrap;
  }}
  #play-btn:hover {{ background: rgba(120,120,200,0.4); }}
  #stats {{ font-size: 0.8rem; color: #6060a0; white-space: nowrap; }}

  #network {{
    width: 100%; height: calc(100vh - 52px);
  }}

  div.vis-navigation div.vis-button {{
    opacity: 0.5; filter: grayscale(1) brightness(1.8);
  }}
  div.vis-navigation div.vis-button:hover {{
    opacity: 1.0; filter: grayscale(1) brightness(2.2);
    box-shadow: 0 0 6px 2px rgba(255,255,255,0.4) !important;
  }}
  div.vis-navigation div.vis-button:active {{
    filter: grayscale(1) brightness(2.5);
    box-shadow: 0 0 3px 3px rgba(255,255,255,0.7) !important;
  }}
</style>
</head>
<body>

<div id="controls">
  <label>Year:</label>
  <input id="year-slider" type="range" min="{y_min}" max="{y_max}" value="{y_min}" step="1">
  <div id="year-display">{y_min}</div>
  <button id="play-btn">&#9654; Play</button>
  <div style="display:flex;gap:0.4rem;margin-left:0.5rem">
    <button class="ctrl-btn" id="btn-zoomin"  title="Zoom in">+</button>
    <button class="ctrl-btn" id="btn-zoomout" title="Zoom out">−</button>
    <button class="ctrl-btn" id="btn-fit"     title="Fit all">&#8982;</button>
  </div>
  <div id="stats">nodes: 0 &nbsp;|&nbsp; edges: 0</div>
</div>
<div id="network"></div>

<script>
const RAW = {graph_data};

// ── vis.js from CDN (inline would be ~1MB; use CDN for keyword graph) ──
const script = document.createElement("script");
script.src = "https://unpkg.com/vis-network@9.1.9/dist/vis-network.min.js";
script.onload = () => init();
document.head.appendChild(script);

function init() {{
  const container = document.getElementById("network");
  const maxW = RAW.max_weight;

  // Deep-copy and set initial hidden state
  const nodesArr = RAW.nodes.map(n => ({{ ...n, hidden: true }}));
  const edgesArr = RAW.edges.map(e => ({{ ...e, hidden: true, width: 1 }}));

  const nodes = new vis.DataSet(nodesArr);
  const edges = new vis.DataSet(edgesArr);

  const network = new vis.Network(container, {{ nodes, edges }}, {{
    physics: false,
    interaction: {{ hover: true, tooltipDelay: 100 }},
    nodes: {{ font: {{ size: 11, color: "white" }}, shape: "dot" }},
    edges: {{ smooth: false }},
  }});

  // ── Update graph for a given year ──
  function updateYear(year) {{
    document.getElementById("year-display").textContent = year;
    document.getElementById("year-slider").value = year;

    const nodeUpdates = [];
    for (const n of RAW.nodes) {{
      nodeUpdates.push({{ id: n.id, hidden: n.first_year > year }});
    }}
    nodes.update(nodeUpdates);

    const edgeUpdates = [];
    let visibleEdges = 0;
    for (const e of RAW.edges) {{
      const w = e.years.filter(y => y <= year).length;
      const hidden = w === 0;
      if (!hidden) visibleEdges++;
      edgeUpdates.push({{
        id: e.id,
        hidden,
        width: hidden ? 1 : Math.max(1, Math.log1p(w) / Math.log1p(maxW) * 14),
        title: hidden ? "" : `co-occurrences: ${{w}}`,
      }});
    }}
    edges.update(edgeUpdates);

    const visibleNodes = RAW.nodes.filter(n => n.first_year <= year).length;
    document.getElementById("stats").textContent =
      `nodes: ${{visibleNodes}} | edges: ${{visibleEdges}}`;
  }}

  // ── Slider ──
  const slider = document.getElementById("year-slider");
  slider.addEventListener("input", () => updateYear(parseInt(slider.value)));
  updateYear({y_min});

  // ── Zoom buttons ──
  document.getElementById("btn-zoomin").addEventListener("click", () => {{
    network.moveTo({{ scale: network.getScale() * 1.3 }});
  }});
  document.getElementById("btn-zoomout").addEventListener("click", () => {{
    network.moveTo({{ scale: network.getScale() / 1.3 }});
  }});
  document.getElementById("btn-fit").addEventListener("click", () => {{
    network.fit({{ animation: {{ duration: 400, easingFunction: "easeInOutQuad" }} }});
  }});

  // ── Play / Pause ──
  let playing = false;
  let timer = null;
  const playBtn = document.getElementById("play-btn");

  playBtn.addEventListener("click", () => {{
    if (playing) {{
      clearInterval(timer);
      playing = false;
      playBtn.innerHTML = "&#9654; Play";
    }} else {{
      let y = parseInt(slider.value);
      if (y >= {y_max}) y = {y_min};
      playing = true;
      playBtn.innerHTML = "&#9646;&#9646; Pause";
      timer = setInterval(() => {{
        y++;
        updateYear(y);
        if (y >= {y_max}) {{
          clearInterval(timer);
          playing = false;
          playBtn.innerHTML = "&#9654; Play";
        }}
      }}, 600);
    }}
  }});
}}
</script>
</body>
</html>"""


if __name__ == "__main__":
    main()
