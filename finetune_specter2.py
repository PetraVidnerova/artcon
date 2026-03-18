"""
Fine-tune SPECTER2 on the ARTCON dataset using bibliographic coupling
as the training signal (papers sharing many references → should embed close).

Method
------
  - Loads the pre-trained SPECTER2 proximity adapter
  - Adds a LoRA adapter on top (trains only ~2M parameters instead of 110M)
  - Trains with MultipleNegativesRankingLoss (in-batch negatives — efficient
    for small datasets, no need to mine hard negatives explicitly)
  - Positive pairs: paper pairs with bibliographic coupling score >= MIN_COUPLING
  - Saves the LoRA adapter weights (small file, ~few MB)

After training
--------------
  Copy the saved adapter directory back to this machine and point
  embed_specter2.py at it (see OUTPUT_ADAPTER path below).

Requirements (GPU machine)
--------------------------
  pip install torch transformers adapters sentence-transformers scipy pandas numpy tqdm

Usage
-----
  python finetune_specter2.py
  python finetune_specter2.py --epochs 5 --lr 3e-5 --min-coupling 3
"""
import argparse
import random
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from adapters import AutoAdapterModel, LoRAConfig
from tqdm import tqdm

# ── Files (paths relative to this script) ──────────────────────────────────
COUPLING_FILE  = "ArtCon_coupling.npz"
PAPERS_FILE    = "ArtCon_clusters.csv"   # has title + abstract columns
THESES_FILE    = "ArtCon_theses.csv"     # fallback for abstract column
OUTPUT_ADAPTER = "specter2_finetuned"    # directory to save adapter weights

# ── Defaults ────────────────────────────────────────────────────────────────
BASE_MODEL     = "allenai/specter2_base"
PROXY_ADAPTER  = "allenai/specter2"      # proximity adapter
LORA_RANK      = 16
LORA_ALPHA     = 32
LORA_DROPOUT   = 0.1
MAX_LENGTH     = 512
BATCH_SIZE     = 16
MIN_COUPLING   = 2     # minimum shared references to count as a positive pair
SEED           = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Dataset ──────────────────────────────────────────────────────────────────

class CouplingPairDataset(Dataset):
    """Each item is a (anchor_text, positive_text) pair.
    Negatives are handled in-batch by MultipleNegativesRankingLoss."""

    def __init__(self, texts: list[str], pairs: list[tuple[int, int]]):
        self.texts = texts
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        i, j = self.pairs[idx]
        return self.texts[i], self.texts[j]


def collate_fn(tokenizer, max_length):
    def _collate(batch):
        anchors, positives = zip(*batch)
        enc_a = tokenizer(list(anchors),   padding=True, truncation=True,
                          max_length=max_length, return_tensors="pt",
                          return_token_type_ids=False)
        enc_p = tokenizer(list(positives), padding=True, truncation=True,
                          max_length=max_length, return_tensors="pt",
                          return_token_type_ids=False)
        return enc_a, enc_p
    return _collate


# ── Loss ─────────────────────────────────────────────────────────────────────

def multiple_negatives_ranking_loss(emb_a: torch.Tensor,
                                    emb_p: torch.Tensor,
                                    scale: float = 20.0) -> torch.Tensor:
    """
    Each anchor is matched to its positive; all other positives in the batch
    serve as negatives.  Equivalent to cross-entropy over cosine similarity.
    scale = temperature^-1 (20 is a common default).
    """
    emb_a = F.normalize(emb_a, dim=-1)
    emb_p = F.normalize(emb_p, dim=-1)
    scores = torch.mm(emb_a, emb_p.T) * scale          # (B, B)
    labels = torch.arange(len(emb_a), device=emb_a.device)
    return F.cross_entropy(scores, labels)


# ── Model helpers ─────────────────────────────────────────────────────────────

def mean_pool(last_hidden_state: torch.Tensor,
              attention_mask: torch.Tensor) -> torch.Tensor:
    """Masked mean pooling (CLS token is fine too; mean pool is slightly better)."""
    mask = attention_mask.unsqueeze(-1).float()
    return (last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


def encode(model, batch: dict, pool: str = "cls") -> torch.Tensor:
    out = model(**batch)
    if pool == "cls":
        return out.last_hidden_state[:, 0, :]
    return mean_pool(out.last_hidden_state, batch["attention_mask"])


# ── Main ──────────────────────────────────────────────────────────────────────

def build_texts(papers_file, theses_file, tokenizer):
    """Build title + SEP + abstract strings for all 696 papers."""
    df = pd.read_csv(papers_file).fillna("")
    if "abstract" not in df.columns:
        df2 = pd.read_csv(theses_file).fillna("")
        df["abstract"] = df2["abstract"].values
    sep = tokenizer.sep_token
    return [
        (row["title"] or "") + sep + (row.get("abstract") or "")
        for _, row in df.iterrows()
    ]


def build_pairs(coupling_npz, min_coupling):
    mat = sp.load_npz(coupling_npz).tocoo()
    pairs = [(int(i), int(j), int(v))
             for i, j, v in zip(mat.row, mat.col, mat.data)
             if i < j and v >= min_coupling]
    # Shuffle and deduplicate (COO may have duplicates)
    seen = set()
    filtered = []
    for i, j, v in pairs:
        key = (i, j)
        if key not in seen:
            seen.add(key)
            filtered.append((i, j))
    random.shuffle(filtered)
    return filtered


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",       type=int,   default=3)
    parser.add_argument("--lr",           type=float, default=2e-5)
    parser.add_argument("--batch-size",   type=int,   default=BATCH_SIZE)
    parser.add_argument("--min-coupling", type=int,   default=MIN_COUPLING)
    parser.add_argument("--lora-rank",    type=int,   default=LORA_RANK)
    parser.add_argument("--pool",         choices=["cls", "mean"], default="cls")
    args = parser.parse_args()

    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cpu":
        print("WARNING: running on CPU will be very slow. Use a GPU machine.")

    # ── Load tokenizer and build texts ──────────────────────────────────────
    print("Loading tokenizer…")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    texts = build_texts(PAPERS_FILE, THESES_FILE, tokenizer)
    print(f"  {len(texts)} paper texts built")

    # ── Build training pairs ─────────────────────────────────────────────────
    pairs = build_pairs(COUPLING_FILE, args.min_coupling)
    print(f"  {len(pairs)} positive pairs (min coupling={args.min_coupling})")

    dataset = CouplingPairDataset(texts, pairs)
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                         collate_fn=collate_fn(tokenizer, MAX_LENGTH))

    # ── Load SPECTER2 + add LoRA adapter ─────────────────────────────────────
    print("Loading SPECTER2…")
    model = AutoAdapterModel.from_pretrained(BASE_MODEL)
    model.load_adapter(PROXY_ADAPTER, source="hf",
                       load_as="specter2_proximity", set_active=True)

    lora_cfg = LoRAConfig(
        r=args.lora_rank,
        alpha=LORA_ALPHA,
        dropout=LORA_DROPOUT,
        use_gating=False,
    )
    model.add_adapter("coupling_lora", config=lora_cfg)
    model.train_adapter("coupling_lora")   # freeze everything except LoRA
    model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {trainable:,} / {total:,} "
          f"({100*trainable/total:.2f}%)")

    # ── Optimiser + scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01,
    )
    total_steps   = len(loader) * args.epochs
    warmup_steps  = total_steps // 10
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\nTraining for {args.epochs} epochs…")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        bar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}")
        for enc_a, enc_p in bar:
            enc_a = {k: v.to(device) for k, v in enc_a.items()}
            enc_p = {k: v.to(device) for k, v in enc_p.items()}

            emb_a = encode(model, enc_a, args.pool)
            emb_p = encode(model, enc_p, args.pool)
            loss  = multiple_negatives_ranking_loss(emb_a, emb_p)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            bar.set_postfix(loss=f"{loss.item():.4f}")

        avg = total_loss / len(loader)
        print(f"  Epoch {epoch} — avg loss: {avg:.4f}")

    # ── Save adapter ──────────────────────────────────────────────────────────
    model.save_adapter(OUTPUT_ADAPTER, "coupling_lora")
    print(f"\nSaved LoRA adapter → {OUTPUT_ADAPTER}/")
    print("To re-embed with the fine-tuned model, load it in embed_specter2.py:")
    print(f'  model.load_adapter("{OUTPUT_ADAPTER}", load_as="coupling_lora", set_active=True)')


if __name__ == "__main__":
    main()
