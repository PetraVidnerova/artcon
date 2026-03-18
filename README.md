# ARTCON Database

Interactive analysis and visualisation of ~696 papers on **artificial consciousness**.

The pipeline converts a `.docx` bibliography into a browsable graph where nodes are papers,
edges encode similarity or keyword overlap, and nodes can be coloured by publication year,
cluster membership, topic, or philosophical stance.

---

## Quick start

```bash
# Install dependencies (requires Python ≥ 3.14)
uv sync

# Open the finished visualisation
xdg-open index.html
```

---

## Pipeline overview

```
ArtCon_with abstracts.docx
        │
        ▼
  convert_to_csv.py ──────────────────────────► ArtCon.csv
        │
        ▼
  find_openalex_ids.py ───────────────────────► ArtCon_openalex.csv
        │
        ├──► embed_specter2.py ───────────────► specter2_embeddings.npy
        │                                       specter2_index.csv
        │
        ├──► extract_keywords.py ────────────► ArtCon_keywords.csv
        │
        ├──► classify_theses.py ─────────────► ArtCon_theses.csv
        │          (uses theses.py)
        │
        ├──► cluster_papers.py ──────────────► ArtCon_clusters.csv
        │
        ├──► topic_model.py ─────────────────► topic_labels.json
        │                                      (updates ArtCon_clusters.csv)
        │
        ├──► fetch_references.py ────────────► ArtCon_references.csv
        │
        ├──► compute_coupling.py ────────────► ArtCon_coupling.npz
        │                                      ArtCon_coupling.csv
        │
        ├──► export_html.py ─────────────────► graph.html
        ├──► export_timeline.py ─────────────► timeline.html
        └──► export_keyword_graph.py ─────────► keyword_graph.html

index.html          embeds graph.html + timeline.html
index_keywords.html embeds keyword_graph.html
```

---

## Scripts

### 1. `convert_to_csv.py` — Parse bibliography

Parses `ArtCon_with abstracts.docx` (APA format with italic title/journal distinction)
into a structured CSV.

```bash
uv run python3 convert_to_csv.py
```

**Output:** `ArtCon.csv`
**Columns:** `authors, year, title, journal, volume, issue, pages, doi, url, abstract`

---

### 2. `find_openalex_ids.py` — Look up OpenAlex IDs

Matches each paper to an OpenAlex work ID via DOI (primary) or title+author search.
Requires an API key in `openalex_api_key.txt`.

```bash
uv run python3 find_openalex_ids.py
```

**Output:** `ArtCon_openalex.csv` — adds `openalex_id`, `match_method` columns

---

### 3. `embed_specter2.py` — Generate SPECTER2 embeddings

Encodes title + abstract for each paper using the
[SPECTER2](https://huggingface.co/allenai/specter2_base) proximity adapter.
Downloads ~500 MB model on first run. GPU recommended.

```bash
uv run python3 embed_specter2.py
```

**Output:** `specter2_embeddings.npy` (696 × 768 float32), `specter2_index.csv`

---

### 4. `extract_keywords.py` — Extract keyphrases

Extracts keyphrases from abstracts using
[`ml6team/keyphrase-extraction-kbir-inspec`](https://huggingface.co/ml6team/keyphrase-extraction-kbir-inspec).

```bash
uv run python3 extract_keywords.py
```

**Output:** `ArtCon_keywords.csv` — adds `keywords` column (semicolon-separated)

---

### 5. `theses.py` — Thesis pair definitions

Defines 5 philosophical thesis pairs (pro/con) used for classification.
Not a runnable script — imported by `classify_theses.py` and `export_html.py`.

```python
THESIS_PAIRS = [
    {"key": "current_possibility",        "pro": "...", "con": "..."},
    {"key": "future_possibility",         ...},
    {"key": "functionalism",              ...},
    {"key": "computational_functionalism",...},
    {"key": "biology",                    ...},
]
```

---

### 6. `classify_theses.py` — Classify papers by thesis stance

Uses zero-shot NLI (`cross-encoder/nli-deberta-v3-large`) to classify each paper
as **supports / opposes / neutral** for each of the 5 thesis pairs.
GPU strongly recommended (slow on CPU).

```bash
uv run python3 classify_theses.py
```

**Output:** `ArtCon_theses.csv` — adds `<key>__stance` and `<key>__score` columns
per thesis pair

---

### 7. `cluster_papers.py` — Cluster papers

Clusters papers using UMAP + HDBSCAN or Louvain community detection.
Five clustering methods, all saved to the same output file.

```bash
uv run python3 cluster_papers.py               # all methods (default)
uv run python3 cluster_papers.py --method embeddings
uv run python3 cluster_papers.py --method keywords
uv run python3 cluster_papers.py --method stance
uv run python3 cluster_papers.py --method louvain_embeddings
uv run python3 cluster_papers.py --method louvain_keywords
```

| Method | Features | Algorithm |
|--------|----------|-----------|
| `embeddings` | SPECTER2 embeddings | UMAP → HDBSCAN |
| `keywords` | TF-IDF on keyphrases | UMAP → HDBSCAN |
| `stance` | Thesis stance vectors (5-dim) | HDBSCAN directly |
| `louvain_embeddings` | Cosine similarity graph | Louvain community detection |
| `louvain_keywords` | Keyword co-occurrence graph | Louvain community detection |

**Output:** `ArtCon_clusters.csv` — adds one `cluster_<method>` column per method

---

### 8. `topic_model.py` — BERTopic topic modelling

Fits [BERTopic](https://maartengr.github.io/BERTopic/) on paper abstracts using
SPECTER2 embeddings. Auto-labels each topic with representative keyphrases
(e.g. *"language, llms, language models"*).

```bash
uv run python3 topic_model.py
```

**Output:**
- `topic_labels.json` — `{topic_id: "label"}` mapping used by the graph
- `ArtCon_clusters.csv` — adds `cluster_topic` column

---

### 9. `fetch_references.py` — Fetch citation data from OpenAlex

Retrieves the reference list for each paper that has an OpenAlex ID.
Supports resuming — re-run safely if interrupted.

```bash
uv run python3 fetch_references.py
```

**Output:** `ArtCon_references.csv` — `openalex_id`, `references` (pipe-separated IDs)

---

### 10. `compute_coupling.py` — Bibliographic coupling matrix

Computes the number of shared references for every pair of papers
(bibliographic coupling). Used as training signal for SPECTER2 fine-tuning.

```bash
uv run python3 compute_coupling.py
```

**Output:**
- `ArtCon_coupling.npz` — sparse 696 × 696 scipy CSR matrix
- `ArtCon_coupling.csv` — human-readable ranked list of coupled pairs

---

### 11. `finetune_specter2.py` — Fine-tune SPECTER2 *(GPU machine)*

Fine-tunes SPECTER2 using bibliographic coupling as the training signal.
Trains only a LoRA adapter (~2M parameters) on top of the frozen base model
to avoid overfitting on the small dataset.

**Copy the whole project directory to the GPU machine, then:**

```bash
pip install torch transformers adapters sentence-transformers scipy pandas numpy tqdm

python finetune_specter2.py                           # defaults (3 epochs, min coupling 2)
python finetune_specter2.py --epochs 5 --min-coupling 3
python finetune_specter2.py --epochs 3 --lr 1e-5 --lora-rank 32
```

**Output:** `specter2_finetuned/` — LoRA adapter weights

After training, copy `specter2_finetuned/` back and re-run `embed_specter2.py`
with the adapter loaded:
```python
model.load_adapter("specter2_finetuned", load_as="coupling_lora", set_active=True)
```
Then re-run the full pipeline from step 7 onwards to update visualisations.

---

### 12. `export_html.py` — Generate main document graph

Generates the interactive document graph with vis.js.
Nodes = papers, coloured by year / cluster / topic / thesis stance.
Edge modes: similarity (cosine) or keyword co-occurrence, switchable in the UI.

```bash
uv run python3 export_html.py
uv run python3 export_html.py --threshold 0.97 --output graph_strict.html
```

**Output:** `graph.html`
**UI controls:**
- Similarity / Keywords toggle (top right)
- Threshold buttons: similarity (0.92–0.98) or min shared keywords (1–3)
- Colour nodes by: year / clusters (5 methods) / topics / none
- Highlight by thesis: colours nodes by philosophical stance

---

### 13. `export_timeline.py` — Generate thesis timeline charts

Generates stacked bar charts showing papers per year aligned with each thesis
(supports / opposes / neutral).

```bash
uv run python3 export_timeline.py
```

**Output:** `timeline.html`

---

### 14. `export_keyword_graph.py` — Generate keyword co-occurrence graph

Generates an interactive keyword graph with a year slider.
Nodes = keywords, edges = number of papers sharing both keywords.

```bash
uv run python3 export_keyword_graph.py
```

**Output:** `keyword_graph.html`

---

### 15. `export_gephi.py` — Export to Gephi

Exports a GEXF file for further analysis in [Gephi](https://gephi.org/).

```bash
uv run python3 export_gephi.py
```

**Output:** `artcon_graph.gexf`

---

## Data files

| File | Description |
|------|-------------|
| `ArtCon_with abstracts.docx` | Original bibliography (source, not generated) |
| `ArtCon.csv` | Parsed bibliography |
| `ArtCon_openalex.csv` | + OpenAlex IDs |
| `ArtCon_keywords.csv` | + extracted keyphrases |
| `ArtCon_theses.csv` | + thesis stance classifications |
| `ArtCon_clusters.csv` | + all cluster assignments |
| `ArtCon_references.csv` | Fetched OpenAlex reference lists |
| `ArtCon_coupling.csv` | Bibliographic coupling pairs (ranked) |
| `ArtCon_coupling.npz` | Bibliographic coupling sparse matrix |
| `specter2_embeddings.npy` | SPECTER2 embeddings (696 × 768) |
| `specter2_index.csv` | Row index → paper metadata mapping |
| `topic_labels.json` | BERTopic topic labels |
| `openalex_api_key.txt` | OpenAlex API key *(not committed)* |

---

## Visualisation pages

| File | Contents |
|------|----------|
| `index.html` | Document graph + thesis timelines |
| `index_keywords.html` | Keyword co-occurrence graph |
| `graph.html` | Document graph (standalone) |
| `timeline.html` | Thesis timeline charts (standalone) |
| `keyword_graph.html` | Keyword graph (standalone) |

---

## Dependencies

Managed with [uv](https://github.com/astral-sh/uv). See `pyproject.toml` for the full list.
Key packages: `adapters`, `bertopic`, `hdbscan`, `networkx`, `pyvis`, `scikit-learn`,
`sentence-transformers`, `torch`, `umap-learn`.

```bash
uv sync        # install all dependencies
uv run <script>  # run any script in the project environment
```
