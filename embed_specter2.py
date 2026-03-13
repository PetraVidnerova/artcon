"""
Generate SPECTER2 embeddings for each entry in ArtCon.csv (or ArtCon_openalex.csv).
Input:  title + abstract per row
Output: embeddings saved as .npy (shape: N x 768) + index CSV mapping row → entry
"""
import csv
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from adapters import AutoAdapterModel

INPUT      = "ArtCon_openalex.csv"   # falls back to ArtCon.csv if not found
OUTPUT_NPY = "specter2_embeddings.npy"
OUTPUT_IDX = "specter2_index.csv"
BATCH_SIZE = 16
MAX_LENGTH = 512
ADAPTER    = "allenai/specter2"      # proximity adapter — good for similarity search


def load_model():
    print("Loading SPECTER2 model and adapter...")
    tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
    model = AutoAdapterModel.from_pretrained("allenai/specter2_base")
    model.load_adapter(ADAPTER, source="hf", load_as="specter2", set_active=True)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"  Running on: {device}")
    return tokenizer, model, device


def load_rows():
    import os
    path = INPUT if os.path.exists(INPUT) else "ArtCon.csv"
    print(f"Loading entries from {path}...")
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"  {len(rows)} entries loaded")
    return rows


def encode_batch(texts, tokenizer, model, device):
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
        return_token_type_ids=False,
        max_length=MAX_LENGTH,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        output = model(**inputs)
    # CLS token embedding
    return output.last_hidden_state[:, 0, :].cpu().float().numpy()


def main():
    tokenizer, model, device = load_model()
    rows = load_rows()

    # Build input texts: title + SEP + abstract
    texts = [
        (row.get("title") or "") + tokenizer.sep_token + (row.get("abstract") or "")
        for row in rows
    ]

    # Write index CSV
    index_fields = ["row_index", "authors", "year", "title", "doi", "openalex_id"]
    with open(OUTPUT_IDX, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=index_fields, extrasaction="ignore")
        writer.writeheader()
        for i, row in enumerate(rows):
            writer.writerow({"row_index": i, **row})

    # Encode in batches
    all_embeddings = []
    batches = [texts[i:i + BATCH_SIZE] for i in range(0, len(texts), BATCH_SIZE)]

    for batch in tqdm(batches, desc="Embedding", unit="batch"):
        emb = encode_batch(batch, tokenizer, model, device)
        all_embeddings.append(emb)

    embeddings = np.vstack(all_embeddings)
    np.save(OUTPUT_NPY, embeddings)

    print(f"\nDone. Embeddings shape: {embeddings.shape}")
    print(f"  {OUTPUT_NPY}  — float32 array, one row per entry")
    print(f"  {OUTPUT_IDX} — index mapping row number to entry metadata")


if __name__ == "__main__":
    main()
