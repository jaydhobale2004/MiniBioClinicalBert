"""
Embeds all notes and saves the search index to artifacts/search_index/.
Automatically resumes from the last checkpoint if the run is interrupted.
"""

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


MODEL_DIR = "artifacts/sbert_bi_encoder"
IN_JSONL  = Path("data/corpus/synthea_notes.jsonl")
OUT_DIR   = Path("artifacts/search_index")

BATCH_SIZE        = 16
MAX_LEN           = 256
CHECKPOINT_EVERY  = 50  # save progress every 50 batches (~800 notes)

# these files are cleaned up automatically after a successful run
_CKPT_VECS = OUT_DIR / "_ckpt_vecs.npy"
_CKPT_IDX  = OUT_DIR / "_ckpt_idx.txt"


def get_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_notes(jsonl_path: Path) -> List[Dict]:
    notes = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            notes.append(json.loads(line))
    return notes


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask   = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def embed_text_batch(texts: List[str], tokenizer, model, device: torch.device) -> np.ndarray:
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
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-9
    return x / norms


def save_checkpoint(vecs: list, next_batch_idx: int):
    np.save(_CKPT_VECS, np.vstack(vecs))
    _CKPT_IDX.write_text(str(next_batch_idx))


def load_checkpoint() -> tuple[list, int]:
    """Returns (accumulated_vecs, next_batch_index) or ([], 0) if there's no checkpoint."""
    if _CKPT_VECS.exists() and _CKPT_IDX.exists():
        saved = np.load(_CKPT_VECS)
        next_idx = int(_CKPT_IDX.read_text().strip())
        print(f"Resuming from checkpoint: {saved.shape[0]} notes already embedded, "
              f"next batch index = {next_idx}.")
        return [saved], next_idx
    return [], 0


def clear_checkpoint():
    _CKPT_VECS.unlink(missing_ok=True)
    _CKPT_IDX.unlink(missing_ok=True)


def main():
    device = get_device()
    print("Device:", device)

    if not IN_JSONL.exists():
        raise FileNotFoundError(f"Missing {IN_JSONL}. Run make_notes_from_csv.py first.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    out = Path(MODEL_DIR)
    has_model = out.exists() and any(
        f.suffix in (".json", ".bin", ".safetensors") for f in out.iterdir()
    )
    actual_model = MODEL_DIR if has_model else "emilyalsentzer/Bio_ClinicalBERT"
    if not has_model:
        print(f"SBERT model not found at '{MODEL_DIR}'. Falling back to {actual_model}.")
    tokenizer = AutoTokenizer.from_pretrained(actual_model)
    model     = AutoModel.from_pretrained(actual_model).to(device)
    model.eval()

    notes = load_notes(IN_JSONL)
    texts = [n["text"] for n in notes]
    print(f"Loaded {len(texts)} notes.")

    all_vecs, start_batch = load_checkpoint()
    start_note = start_batch * BATCH_SIZE

    if start_note >= len(texts):
        print("Checkpoint already covers all notes — rebuilding final index.")
        all_vecs, start_batch, start_note = [], 0, 0

    for batch_i, i in enumerate(range(start_note, len(texts), BATCH_SIZE), start=start_batch):
        batch = texts[i : i + BATCH_SIZE]
        try:
            vecs = embed_text_batch(batch, tokenizer, model, device)
            all_vecs.append(vecs)
        except Exception as e:
            print(f"\n[ERROR] Batch {batch_i} (notes {i}-{i+len(batch)-1}) failed: {e}")
            print("Saving checkpoint and exiting. Re-run to resume.")
            save_checkpoint(all_vecs, batch_i)
            raise

        current_batch = batch_i + 1
        if current_batch % CHECKPOINT_EVERY == 0:
            save_checkpoint(all_vecs, current_batch)

        if current_batch % 50 == 0 or (i + BATCH_SIZE) >= len(texts):
            done = min(i + BATCH_SIZE, len(texts))
            print(f"Embedded {done}/{len(texts)}  ({100*done/len(texts):.1f}%)")

    embeddings = normalize_rows(np.vstack(all_vecs))

    np.save(OUT_DIR / "embeddings.npy", embeddings)
    with (OUT_DIR / "meta.jsonl").open("w", encoding="utf-8") as f:
        for n in notes:
            f.write(json.dumps(n, ensure_ascii=False) + "\n")

    clear_checkpoint()
    print(f"\nDone. Saved {len(embeddings)} embeddings to:")
    print(" -", OUT_DIR / "embeddings.npy")
    print(" -", OUT_DIR / "meta.jsonl")


if __name__ == "__main__":
    main()
