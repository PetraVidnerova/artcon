"""
Generate a standalone timeline HTML chart showing, for each thesis pair,
the number of papers aligned (supports / opposes / neutral) per year.

Usage:
    uv run python3 export_timeline.py
"""
import csv
import json
from collections import defaultdict

THESES_FILE = "ArtCon_theses.csv"
OUTPUT_FILE = "timeline.html"

THESIS_PAIRS = [
    {"key": "current_possibility",        "label": "Currently possible?"},
    {"key": "future_possibility",         "label": "Future possibility?"},
    {"key": "functionalism",              "label": "Functionalism?"},
    {"key": "computational_functionalism","label": "Computational functionalism?"},
    {"key": "biology",                    "label": "Biology necessary?"},
]


def build_chart_data():
    rows = []
    with open(THESES_FILE, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    years = sorted(
        set(int(r["year"]) for r in rows if r.get("year", "").strip().isdigit())
    )

    charts = []
    for pair in THESIS_PAIRS:
        key = pair["key"]
        col = f"{key}__stance"
        counts = defaultdict(lambda: {"supports": 0, "opposes": 0, "neutral": 0})
        for row in rows:
            yr = row.get("year", "").strip()
            if not yr.isdigit():
                continue
            stance = row.get(col, "neutral").strip() or "neutral"
            if stance not in counts[int(yr)]:
                stance = "neutral"
            counts[int(yr)][stance] += 1
        charts.append({
            "key": key,
            "label": pair["label"],
            "years": years,
            "supports": [counts[y]["supports"] for y in years],
            "opposes":  [counts[y]["opposes"]  for y in years],
            "neutral":  [counts[y]["neutral"]   for y in years],
        })
    return charts


def render_html(charts):
    charts_json = json.dumps(charts, indent=2)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ARTCON – Thesis Timelines</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: "Segoe UI", system-ui, sans-serif;
    background: #0e1117;
    color: #e0e0e0;
    padding: 2rem 3rem 3rem;
  }}
  h2 {{
    font-size: 1.3rem;
    font-weight: 600;
    color: #ffffff;
    margin-bottom: 0.4rem;
  }}
  .subtitle {{
    font-size: 0.88rem;
    color: #9090a0;
    margin-bottom: 2rem;
  }}
  .grid {{
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 2rem;
  }}
  .card {{
    background: #13151f;
    border: 1px solid #2a2a3a;
    border-radius: 8px;
    padding: 1.2rem 1.4rem 1rem;
  }}
  .card:last-child:nth-child(odd) {{
    grid-column: 1 / -1;
    max-width: 50%;
    justify-self: center;
  }}
  .card-title {{
    font-size: 0.9rem;
    font-weight: 600;
    color: #c8c8d8;
    margin-bottom: 0.8rem;
    text-align: center;
  }}
  canvas {{ width: 100% !important; }}
</style>
</head>
<body>

<h2>Thesis Timelines</h2>
<p class="subtitle">
  Number of papers per year aligned with each thesis, by stance
  (supports&thinsp;/&thinsp;opposes&thinsp;/&thinsp;neutral).
</p>

<div class="grid" id="grid"></div>

<script>
const CHARTS = {charts_json};

const COLORS = {{
  supports: {{ fill: "rgba(70,180,130,0.75)",  border: "rgba(70,180,130,1)"  }},
  opposes:  {{ fill: "rgba(220,80,80,0.75)",   border: "rgba(220,80,80,1)"   }},
  neutral:  {{ fill: "rgba(120,120,150,0.55)", border: "rgba(120,120,150,1)" }},
}};

const LABELS = {{
  supports: "supports",
  opposes:  "opposes",
  neutral:  "neutral",
}};

const grid = document.getElementById("grid");

CHARTS.forEach(c => {{
  const card = document.createElement("div");
  card.className = "card";
  card.innerHTML = `<div class="card-title">${{c.label}}</div><canvas id="chart-${{c.key}}"></canvas>`;
  grid.appendChild(card);

  const ctx = document.getElementById("chart-" + c.key).getContext("2d");
  new Chart(ctx, {{
    type: "bar",
    data: {{
      labels: c.years,
      datasets: ["supports","opposes","neutral"].map(s => ({{
        label: LABELS[s],
        data: c[s],
        backgroundColor: COLORS[s].fill,
        borderColor: COLORS[s].border,
        borderWidth: 1,
      }})),
    }},
    options: {{
      responsive: true,
      plugins: {{
        legend: {{
          position: "bottom",
          labels: {{ color: "#9090a0", boxWidth: 12, padding: 10, font: {{ size: 11 }} }},
        }},
        tooltip: {{ mode: "index", intersect: false }},
      }},
      scales: {{
        x: {{
          stacked: true,
          ticks: {{ color: "#9090a0", maxRotation: 45, font: {{ size: 10 }} }},
          grid: {{ color: "#1e2030" }},
        }},
        y: {{
          stacked: true,
          ticks: {{ color: "#9090a0", stepSize: 1, font: {{ size: 10 }} }},
          grid: {{ color: "#1e2030" }},
          title: {{ display: false }},
        }},
      }},
    }},
  }});
}});
</script>
</body>
</html>
"""


def main():
    charts = build_chart_data()
    html = render_html(charts)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(html)
    total = sum(sum(c["supports"]) + sum(c["opposes"]) + sum(c["neutral"]) for c in charts)
    print(f"Written {OUTPUT_FILE}  ({len(charts)} charts, data over {len(charts[0]['years'])} years)")


if __name__ == "__main__":
    main()
