"""
Convert ArtCon_from_zotero.csv to ArtCon.csv.
Maps Zotero export columns to the project's canonical field names.
"""
import csv

ZOTERO_IN = '/home/petra/work_trust/ARTCON_DATABASE/ArtCon_from_zotero.csv'
CSV_OUT   = '/home/petra/work_trust/ARTCON_DATABASE/ArtCon.csv'

FIELDS = ['authors', 'year', 'title', 'journal', 'volume', 'issue', 'pages', 'doi', 'url', 'abstract']


def convert(row):
    doi = row.get('DOI', '').strip()
    url = row.get('Url', '').strip()
    return {
        'authors': row.get('Author', '').strip(),
        'year':    row.get('Publication Year', '').strip(),
        'title':   row.get('Title', '').strip(),
        'journal': row.get('Publication Title', '').strip(),
        'volume':  row.get('Volume', '').strip(),
        'issue':   row.get('Issue', '').strip(),
        'pages':   row.get('Pages', '').strip(),
        'doi':     doi,
        'url':     url,
        'abstract': row.get('Abstract Note', '').strip(),
    }


def main():
    with open(ZOTERO_IN, newline='', encoding='utf-8') as fin:
        reader = csv.DictReader(fin)
        rows = [convert(r) for r in reader]

    with open(CSV_OUT, 'w', newline='', encoding='utf-8') as fout:
        writer = csv.DictWriter(fout, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f'Written {len(rows)} rows to {CSV_OUT}')


if __name__ == '__main__':
    main()
