"""
Find OpenAlex IDs for each entry in ArtCon.csv.
Strategy:
  1. If DOI present → look up directly by DOI (reliable)
  2. Else → search by title, verify with author/year match
Output: ArtCon_openalex.csv with added columns openalex_id, match_method
"""
import csv
import time
import re
import requests
from tqdm import tqdm

with open("openalex_api_key.txt") as _f:
    API_KEY = _f.read().strip()

EMAIL = "petra@cs.cas.cz"
BASE = "https://api.openalex.org"
HEADERS = {"User-Agent": f"ArtConBot/1.0 (mailto:{EMAIL})"}

INPUT  = "ArtCon.csv"
OUTPUT = "ArtCon_openalex.csv"


def get_by_doi(doi: str) -> str | None:
    """Return OpenAlex ID for a DOI, or None."""
    url = f"{BASE}/works/https://doi.org/{doi}"
    r = requests.get(url, headers=HEADERS, params={"api_key": API_KEY}, timeout=10)
    if r.status_code == 200:
        return r.json().get("id", "")
    return None


def search_by_title(title: str, year: str, authors: str) -> tuple[str, str]:
    """
    Search OpenAlex by title. Return (openalex_id, match_method) or ('', '').
    Tries exact title match first, then fuzzy.
    """
    # Clean title for search
    title_clean = re.sub(r"['\u2018\u2019\u201c\u201d]", "", title).strip()

    params = {
        "filter": f"display_name.search:{title_clean}",
        "select": "id,display_name,publication_year,authorships",
        "per_page": 5,
        "api_key": API_KEY,
    }
    r = requests.get(f"{BASE}/works", headers=HEADERS, params=params, timeout=10)
    if r.status_code != 200:
        return "", ""

    results = r.json().get("results", [])
    if not results:
        return "", ""

    # Score candidates
    title_lower = title_clean.lower()
    year_int = int(year) if year.isdigit() else None

    # Extract first author surname for matching
    first_author_surname = ""
    if authors:
        first_author_surname = authors.split(",")[0].strip().lower()

    best_id = ""
    best_method = ""

    for work in results:
        cand_title = (work.get("display_name") or "").lower()
        cand_year = work.get("publication_year")

        # Check title similarity (simple: one contains the other, or close enough)
        title_match = (
            title_lower == cand_title
            or title_lower in cand_title
            or cand_title in title_lower
        )

        year_match = (year_int is None) or (cand_year == year_int)

        # Check first author
        author_match = False
        if first_author_surname:
            for auth in work.get("authorships", []):
                name = (auth.get("author", {}).get("display_name") or "").lower()
                if first_author_surname in name:
                    author_match = True
                    break
        else:
            author_match = True  # no author to check

        if title_match and year_match and author_match:
            best_id = work["id"]
            best_method = "title+year+author"
            break
        elif title_match and year_match and not best_id:
            best_id = work["id"]
            best_method = "title+year"
        elif title_match and author_match and not best_id:
            best_id = work["id"]
            best_method = "title+author"
        elif title_match and not best_id:
            best_id = work["id"]
            best_method = "title"

    return best_id, best_method


def main():
    with open(INPUT, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    fieldnames = list(rows[0].keys()) + ["openalex_id", "match_method"]

    found = 0
    with open(OUTPUT, "w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=fieldnames)
        writer.writeheader()

        bar = tqdm(rows, unit="entry", desc="Looking up")
        for row in bar:
            doi = row.get("doi", "").strip()
            title = row.get("title", "").strip()
            year = row.get("year", "").strip()
            authors = row.get("authors", "").strip()

            openalex_id = ""
            match_method = ""

            if doi:
                openalex_id = get_by_doi(doi) or ""
                if openalex_id:
                    match_method = "doi"
                    found += 1

            if not openalex_id and title:
                openalex_id, match_method = search_by_title(title, year, authors)
                if openalex_id:
                    found += 1

            row["openalex_id"] = openalex_id
            row["match_method"] = match_method
            writer.writerow(row)

            bar.set_postfix(found=found, method=match_method or "-")

            # Polite rate limit: ~10 req/s max, OpenAlex allows more but be nice
            time.sleep(0.1)

    print(f"\nDone. {found}/{len(rows)} entries matched.")
    print(f"Output: {OUTPUT}")


if __name__ == "__main__":
    main()
