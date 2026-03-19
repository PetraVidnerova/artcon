"""
Generate one HTML page per paper in the papers/ subdirectory.

Usage:
    uv run python3 export_paper_pages.py
"""
import colorsys
import json
import os
import pandas as pd

CLUSTERS_FILE        = "ArtCon_clusters.csv"
THESES_FILE          = "ArtCon_theses.csv"
TOPIC_LABELS_FILE    = "topic_labels.json"
TOPIC_LABELS_FT_FILE = "topic_labels_ft.json"
OUTPUT_DIR           = "papers"

CLUSTER_COLS = [
    ("cluster_embeddings",           "Embeddings (HDBSCAN)"),
    ("cluster_keywords",             "Keywords (HDBSCAN)"),
    ("cluster_stance",               "Stance (HDBSCAN)"),
    ("cluster_louvain_embeddings",   "Louvain / embeddings"),
    ("cluster_louvain_keywords",     "Louvain / keywords"),
    ("cluster_topic",                "BERTopic"),
    ("cluster_embeddings_ft",        "Embeddings FT (HDBSCAN)"),
    ("cluster_louvain_embeddings_ft","Louvain / embeddings FT"),
    ("cluster_topic_ft",             "BERTopic (fine-tuned)"),
]

THESIS_KEYS = [
    ("current_possibility",         "AI is currently conscious"),
    ("future_possibility",          "AI could be conscious in future"),
    ("functionalism",               "Functionalism"),
    ("computational_functionalism", "Computational functionalism"),
    ("biology",                     "Biology required for consciousness"),
]

STANCE_STYLE = {
    "supports": ("background:#44dd88;border-color:#22bb66", "supports"),
    "opposes":  ("background:#ff4455;border-color:#cc2233", "opposes"),
    "neutral":  ("background:#333348;border-color:#444460", "neutral"),
}

PAGE_CSS = """
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: #0e1117; color: #c8c8d8;
      font-family: "Segoe UI", system-ui, sans-serif;
      font-size: 0.95rem; line-height: 1.65;
      padding: 0 16px 64px;
    }
    nav {
      position: sticky; top: 0;
      background: rgba(14,17,23,0.97);
      border-bottom: 1px solid #1e2030;
      backdrop-filter: blur(6px);
      padding: 10px 20px;
      display: flex; align-items: center; gap: 12px;
      font-size: 0.82rem; color: #7070a0;
      z-index: 100;
    }
    nav a { color: #8888b8; text-decoration: none; }
    nav a:hover { color: #c0c0e0; text-decoration: underline; }
    nav .disabled { color: #2a2a4a; cursor: default; }
    nav .spacer { flex: 1; }
    nav .counter { color: #4a4a7a; }
    main { max-width: 800px; margin: 36px auto; }
    h1 { font-size: 1.3rem; font-weight: 600; color: #e0e0f0;
         line-height: 1.45; margin-bottom: 10px; }
    .meta { color: #6868a0; font-size: 0.85rem; margin-bottom: 3px; }
    .doi { font-size: 0.83rem; margin: 6px 0 28px; }
    .doi a { color: #5577cc; text-decoration: none; }
    .doi a:hover { text-decoration: underline; }
    section { margin-top: 30px; }
    section h2 {
      font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.09em;
      color: #4a4a90; margin-bottom: 10px; padding-bottom: 6px;
      border-bottom: 1px solid #1a1c2c;
    }
    .abstract p { color: #aaaacc; line-height: 1.8; }
    table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
    td { padding: 6px 8px; border-bottom: 1px solid #16182a; vertical-align: middle; }
    td:first-child { color: #6060a0; width: 52%; }
    .badge {
      display: inline-block; padding: 2px 11px; border-radius: 12px;
      font-size: 0.77rem; font-weight: 500; color: #fff;
    }
    .stance-badge {
      display: inline-block; padding: 2px 11px; border-radius: 12px;
      font-size: 0.77rem; font-weight: 500; color: #fff;
      border: 1px solid transparent;
    }
    .noise { background: #333345 !important; color: #8888aa !important; }
"""


def cluster_to_color(cid: int) -> str:
    if cid < 0:
        return "#333345"
    hue = (cid * 137.508) % 360
    r, g, b = colorsys.hls_to_rgb(hue / 360, 0.55, 0.70)
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"


def build_page(i: int, row, thesis_row, n_total: int, cluster_labels: dict) -> str:
    title    = str(row.get("title", "")   or "").strip() or f"Paper {i}"
    authors  = str(row.get("authors", "") or "").strip()
    year     = str(row.get("year", "")    or "").strip()
    doi      = str(row.get("doi", "")     or "").strip()
    abstract = str((thesis_row or {}).get("abstract", "") or "").strip()

    # ── cluster table ──
    c_rows = []
    for col, label in CLUSTER_COLS:
        if col not in row.index:
            continue
        raw = row[col]
        cid = int(raw) if pd.notna(raw) else -1
        color = cluster_to_color(cid)
        if cid < 0:
            cid_label = "Noise / unclustered"
            badge_cls = "badge noise"
        else:
            default = f"Cluster {cid}"
            raw_label = cluster_labels.get(col, {}).get(str(cid), default)
            # BERTopic columns: prefix with topic id
            if "topic" in col:
                cid_label = f"T{cid}: {raw_label}"
            else:
                cid_label = raw_label
            badge_cls = "badge"
        c_rows.append(
            f'<tr><td>{label}</td>'
            f'<td><span class="{badge_cls}" style="background:{color}">{cid_label}</span></td></tr>'
        )
    clusters_html = "\n".join(c_rows) if c_rows else '<tr><td colspan="2">No cluster data</td></tr>'

    # ── stance table ──
    s_rows = []
    if thesis_row:
        for key, label in THESIS_KEYS:
            stance = str(thesis_row.get(f"{key}__stance", "") or "").strip()
            score  = thesis_row.get(f"{key}__score", "")
            if not stance:
                continue
            style, _ = STANCE_STYLE.get(stance, STANCE_STYLE["neutral"])
            try:
                score_str = f" ({float(score):.2f})" if score and float(score) != 0 else ""
            except (ValueError, TypeError):
                score_str = ""
            s_rows.append(
                f'<tr><td>{label}</td>'
                f'<td><span class="stance-badge" style="{style};color:#fff">'
                f'{stance}{score_str}</span></td></tr>'
            )
    stances_html = "\n".join(s_rows) if s_rows else '<tr><td colspan="2">No stance data</td></tr>'

    doi_link = (f'<a href="https://doi.org/{doi}" target="_blank">{doi}</a>'
                if doi else "—")
    abstract_html = (f"<p>{abstract}</p>"
                     if abstract else "<p><em>No abstract available.</em></p>")

    prev_link = (f'<a href="{i-1}.html">← {i-1}</a>'
                 if i > 0 else '<span class="disabled">←</span>')
    next_link = (f'<a href="{i+1}.html">{i+1} →</a>'
                 if i < n_total - 1 else '<span class="disabled">→</span>')

    escaped_title = title.replace("<", "&lt;").replace(">", "&gt;")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escaped_title}</title>
  <style>{PAGE_CSS}</style>
</head>
<body>
  <nav>
    {prev_link}
    <span class="counter">Paper {i + 1} of {n_total}</span>
    {next_link}
    <span class="spacer"></span>
    <a href="../graph.html">← Back to graph</a>
  </nav>
  <main>
    <h1>{escaped_title}</h1>
    <p class="meta">{authors}</p>
    <p class="meta">Year: {year}</p>
    <p class="doi">DOI: {doi_link}</p>

    <section class="abstract">
      <h2>Abstract</h2>
      {abstract_html}
    </section>

    <section>
      <h2>Cluster assignments</h2>
      <table>{clusters_html}</table>
    </section>

    <section>
      <h2>Thesis stances</h2>
      <table>{stances_html}</table>
    </section>
  </main>
</body>
</html>"""


def main():
    print("Loading data…")
    df = pd.read_csv(CLUSTERS_FILE)
    df_theses = pd.read_csv(THESES_FILE).fillna("") if os.path.exists(THESES_FILE) else None

    # Load topic labels for cluster badge text
    cluster_labels: dict[str, dict] = {}
    for col, fname in [("cluster_topic", TOPIC_LABELS_FILE),
                       ("cluster_topic_ft", TOPIC_LABELS_FT_FILE)]:
        if os.path.exists(fname):
            with open(fname, encoding="utf-8") as f:
                cluster_labels[col] = json.load(f)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    n = len(df)

    for i, row in df.iterrows():
        thesis_row = (df_theses.iloc[i].to_dict()
                      if df_theses is not None and i < len(df_theses) else None)
        html = build_page(i, row, thesis_row, n, cluster_labels)
        with open(os.path.join(OUTPUT_DIR, f"{i}.html"), "w", encoding="utf-8") as f:
            f.write(html)

    print(f"Written {n} pages → {OUTPUT_DIR}/")
    print("Open papers/0.html or link from graph.html")


if __name__ == "__main__":
    main()
