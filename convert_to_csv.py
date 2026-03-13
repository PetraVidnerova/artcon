"""
Convert ArtCon_with abstracts.docx to CSV.
Uses italic run formatting to split title from journal name (APA style).
"""
import re
import csv
import docx


def parse_entry(para):
    """Parse a single bibliography paragraph into fields."""
    # Split citation from abstract at first newline
    full_text = para.text
    if '\n' in full_text:
        citation_text, abstract = full_text.split('\n', 1)
        abstract = abstract.strip()
    else:
        citation_text = full_text
        abstract = ''

    # --- Extract year ---
    year_match = re.search(r'\((\d{4}[a-z]?)\)', citation_text)
    year = year_match.group(1) if year_match else ''

    # --- Extract authors (everything before first year parenthesis) ---
    if year_match:
        authors = citation_text[:year_match.start()].strip().rstrip(',').strip()
    else:
        authors = ''

    # --- Extract DOI ---
    doi_match = re.search(r'https://doi\.org/(\S+?)\.?\s*$', citation_text)
    doi = doi_match.group(1) if doi_match else ''

    # --- Extract non-DOI URL ---
    url = ''
    if not doi_match:
        url_match = re.search(r'https?://\S+', citation_text)
        if url_match:
            url = url_match.group(0).rstrip('.')

    # --- Use run-level italic formatting to extract title and journal ---
    # Rebuild the citation portion run by run
    # Title = non-italic text after "(Year). "
    # Journal = first italic span
    title = ''
    journal = ''
    source_info = ''  # everything after journal (volume, issue, pages, etc.)

    if year_match:
        # Find the character offset in para.text where the title starts
        title_start_in_full = year_match.end() + 2  # skip "). "

        # Walk runs to find italic boundary
        pos = 0
        after_year = False
        title_runs = []
        journal_runs = []
        after_journal = False
        source_runs = []

        for run in para.runs:
            run_text = run.text
            run_end = pos + len(run_text)

            if not after_year:
                # Check if this run contains the end of "(Year). "
                run_in_full_start = pos
                run_in_full_end = run_end
                if run_in_full_end > title_start_in_full:
                    # Title starts within this run
                    after_year = True
                    leftover = run_text[title_start_in_full - pos:]
                    if run.italic:
                        journal_runs.append(leftover)
                    else:
                        title_runs.append(leftover)
            elif not after_journal:
                if run.italic:
                    # We're in journal section
                    if title_runs:
                        # First italic after non-italic title → journal starts here
                        journal_runs.append(run_text)
                    else:
                        journal_runs.append(run_text)
                else:
                    if journal_runs:
                        # Non-italic after journal → source info starts
                        after_journal = True
                        source_runs.append(run_text)
                    else:
                        title_runs.append(run_text)
            else:
                source_runs.append(run_text)

            pos = run_end

        title = ''.join(title_runs).strip().rstrip('.')
        journal = ''.join(journal_runs).strip()
        source_info = ''.join(source_runs).strip()

        # Trim abstract from source_info (source_info may contain '\n...')
        if '\n' in source_info:
            source_info = source_info[:source_info.index('\n')].strip()
    else:
        # Fallback: just use full citation text as title
        title = citation_text.strip()

    # --- Parse volume, issue, pages from source_info ---
    # Format: ", vol(issue), pages. URL" or similar
    # Strip trailing DOI/URL
    source_clean = re.sub(r'https?://\S+', '', source_info).strip().rstrip('.')

    volume = ''
    issue = ''
    pages = ''

    # Match ", volume(issue), pages" pattern
    vol_match = re.match(r',?\s*(\w[\w\-\.]*)\(([^)]+)\),?\s*([\d\–\-–]+(?:–[\d]+)?)?', source_clean)
    if vol_match:
        volume = vol_match.group(1).strip()
        issue = vol_match.group(2).strip()
        pages = (vol_match.group(3) or '').strip()
    else:
        # Try just ", pages" without volume/issue
        pages_match = re.match(r',?\s*([\d]+\s*[–\-]\s*[\d]+)', source_clean)
        if pages_match:
            pages = pages_match.group(1).strip()

    return {
        'authors': authors,
        'year': year,
        'title': title,
        'journal': journal,
        'volume': volume,
        'issue': issue,
        'pages': pages,
        'doi': doi,
        'url': url,
        'abstract': abstract,
    }


def main():
    doc = docx.Document('/home/petra/work_trust/ARTCON_DATABASE/ArtCon_with abstracts.docx')
    paras = [p for p in doc.paragraphs if p.text.strip()]

    fields = ['authors', 'year', 'title', 'journal', 'volume', 'issue', 'pages', 'doi', 'url', 'abstract']
    out_path = '/home/petra/work_trust/ARTCON_DATABASE/ArtCon.csv'

    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for para in paras:
            row = parse_entry(para)
            writer.writerow(row)

    print(f'Written {len(paras)} rows to {out_path}')


if __name__ == '__main__':
    main()
