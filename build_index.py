"""
build_index.py

Goal:
- Read note-like documents from: data/corpus/synthea_notes.jsonl
- Embed each note with MiniBioBERT (encoder)
- Save:
  - artifacts/search_index/embeddings.npy  (float32 vectors, normalized)
  - artifacts/search_index/meta.jsonl      (patient_id, encounter_id, text)

Why:
- Streamlit UI will do semantic search by comparing query embedding to note embeddings.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


# -------------------------
# Configuration
# -------------------------
MODEL_DIR = Path("artifacts/mini_biobert_mlm")           # Your trained model folder
IN_JSONL = Path("data/corpus/synthea_notes.jsonl")       # Notes corpus (JSONL)
OUT_DIR = Path("artifacts/search_index")                 # Where we save the index

BATCH_SIZE = 16
MAX_LEN = 256


def get_device() -> torch.device:
    """Select GPU if available, otherwise CPU."""
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_notes(jsonl_path: Path) -> List[Dict]:
    """
    Load JSONL file where each line is:
      {"encounter_id": ..., "patient_id": ..., "text": ...}
    """
    notes = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            notes.append(json.loads(line))
    return notes


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Convert token embeddings -> single embedding per text using masked mean pooling.

    last_hidden_state: (batch, seq_len, hidden_dim)
    attention_mask:    (batch, seq_len)
    """
    mask = attention_mask.unsqueeze(-1).float()  # (batch, seq, 1)
    summed = (last_hidden_state * mask).sum(dim=1)  # (batch, hidden)
    counts = mask.sum(dim=1).clamp(min=1e-9)        # (batch, 1)
    return summed / counts


def embed_text_batch(
    texts: List[str],
    tokenizer,
    model,
    device: torch.device
) -> np.ndarray:
    """
    Embed a batch of texts and return (batch_size, hidden_dim) numpy array.
    """
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LEN,
        padding=True,
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded)
        vecs = mean_pool(outputs.last_hidden_state, encoded["attention_mask"])

    return vecs.cpu().numpy().astype(np.float32)


def normalize_rows(x: np.ndarray) -> np.ndarray:
    """L2-normalize each row so dot-product equals cosine similarity."""
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-9
    return x / norms


def main():
    device = get_device()
    print("Device:", device)

    if not IN_JSONL.exists():
        raise FileNotFoundError(f"Missing {IN_JSONL}. Run make_notes_from_csv.py first.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModel.from_pretrained(MODEL_DIR).to(device)
    model.eval()

    # Load notes
    notes = load_notes(IN_JSONL)
    texts = [n["text"] for n in notes]
    print(f"Loaded {len(texts)} notes.")

    # Embed notes in batches
    all_vecs = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        vecs = embed_text_batch(batch, tokenizer, model, device)
        all_vecs.append(vecs)
        if (i // BATCH_SIZE) % 50 == 0:
            print(f"Embedded {min(i + BATCH_SIZE, len(texts))}/{len(texts)}")

    embeddings = np.vstack(all_vecs)  # (N, hidden_dim)
    embeddings = normalize_rows(embeddings)

    # Save embeddings
    np.save(OUT_DIR / "embeddings.npy", embeddings)

    # Save metadata (including text for evidence display)
    with (OUT_DIR / "meta.jsonl").open("w", encoding="utf-8") as f:
        for n in notes:
            f.write(json.dumps(n, ensure_ascii=False) + "\n")

    print("Saved index to:")
    print(" -", OUT_DIR / "embeddings.npy")
    print(" -", OUT_DIR / "meta.jsonl")


if __name__ == "__main__":
    main()
