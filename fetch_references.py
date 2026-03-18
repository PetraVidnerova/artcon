"""
Fetch referenced_works for each paper that has an OpenAlex ID.

Output: ArtCon_references.csv
  columns: openalex_id, references (pipe-separated list of referenced OpenAlex IDs)

Usage:
    uv run python3 fetch_references.py
"""
import csv
import time
import requests
import pandas as pd
from tqdm import tqdm

with open("openalex_api_key.txt") as f:
    API_KEY = f.read().strip()

EMAIL   = "petra@cs.cas.cz"
BASE    = "https://api.openalex.org"
HEADERS = {"User-Agent": f"ArtConBot/1.0 (mailto:{EMAIL})"}

INPUT   = "ArtCon_clusters.csv"
OUTPUT  = "ArtCon_references.csv"

DELAY   = 0.1   # seconds between requests (polite pool allows ~10 req/s)


def fetch_references(openalex_id: str) -> list[str]:
    """Return list of referenced OpenAlex work IDs, or [] on failure."""
    # Extract just the work ID (W1234567) from the full URL
    work_id = openalex_id.rstrip("/").split("/")[-1]
    url = f"{BASE}/works/{work_id}"
    try:
        r = requests.get(url, headers=HEADERS,
                         params={"select": "id,referenced_works", "api_key": API_KEY},
                         timeout=15)
        if r.status_code == 200:
            return r.json().get("referenced_works") or []
        return []
    except Exception:
        return []


def main():
    df = pd.read_csv(INPUT).fillna("")
    has_id = df[df["openalex_id"].str.startswith("https://openalex.org/")]
    print(f"Fetching references for {len(has_id)} papers…")

    # Resume support: skip already-fetched IDs
    done = {}
    try:
        existing = pd.read_csv(OUTPUT)
        done = dict(zip(existing["openalex_id"], existing["references"].fillna("")))
        print(f"  Resuming — {len(done)} already fetched")
    except FileNotFoundError:
        pass

    rows = []
    for _, row in tqdm(has_id.iterrows(), total=len(has_id)):
        oid = row["openalex_id"]
        if oid in done:
            rows.append({"openalex_id": oid, "references": done[oid]})
            continue
        refs = fetch_references(oid)
        rows.append({"openalex_id": oid, "references": "|".join(refs)})
        time.sleep(DELAY)

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT, index=False)
    n_with_refs = (out["references"].str.len() > 0).sum()
    total_refs  = out["references"].str.count(r"\|").add(1).where(out["references"] != "", 0).sum()
    print(f"\nWritten {OUTPUT}")
    print(f"  {n_with_refs} papers have references, {int(total_refs)} total reference links")


if __name__ == "__main__":
    main()
