# MiniBioClinicalBERT

A clinical semantic search system built on top of **Bio_ClinicalBERT**, fine-tuned using SimCSE contrastive learning on synthetic Synthea patient notes. Given a patient ID and encounter ID, it answers natural-language questions about that patient's demographics, diagnoses, medications, procedures, and lab results.

---

## What's inside

| Path | Description |
|------|-------------|
| `app.py` | Streamlit search UI — the main app |
| `build_index.py` | Embeds all notes and saves the search index |
| `evaluate.py` | Extraction accuracy evaluation |
| `artifacts/finetune_sbert.py` | SimCSE fine-tuning script |
| `artifacts/make_notes_from_csv.py` | Converts Synthea CSVs into clinical notes |
| `artifacts/sbert_bi_encoder/` | Fine-tuned BERT encoder weights |
| `artifacts/search_index/` | Pre-built embeddings + metadata |
| `data/synthea_csv/` | Raw Synthea CSV tables |
| `data/corpus/` | Generated clinical notes (JSONL + plain text) |

---

## How to download

**Option A — Git clone:**
```bash
git clone https://huggingface.co/jaydhobale/MiniBioClinicalBERT
cd MiniBioClinicalBERT
```

**Option B — Python (no git required):**
```bash
pip install huggingface_hub
```
```python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="jaydhobale/MiniBioClinicalBERT", local_dir="MiniBioClinicalBERT")
cd MiniBioClinicalBERT
```

---

## How to run the app — step by step

### Step 1 — Make sure Python is installed

You need Python 3.9 or higher. Check with:
```bash
python --version
```

### Step 2 — Install all dependencies

```bash
pip install torch transformers streamlit rank_bm25 datasets numpy pandas
```

> If you have a GPU, install the CUDA version of PyTorch from https://pytorch.org for faster performance.

### Step 3 — The index and model are already included

The repo already contains:
- `artifacts/sbert_bi_encoder/` — the fine-tuned encoder (ready to use)
- `artifacts/search_index/embeddings.npy` — pre-built note embeddings
- `artifacts/search_index/meta.jsonl` — note metadata

You do **not** need to run `make_notes_from_csv.py`, `finetune_sbert.py`, or `build_index.py` unless you want to retrain from scratch.

### Step 4 — Launch the app

```bash
streamlit run app.py
```

Your browser will open automatically at `http://localhost:8501`.

### Step 5 — Use the app

1. In the **left sidebar**, paste a **Patient ID** from `data/synthea_csv/patients.csv` (the `Id` column)
2. In the **left sidebar**, paste an **Encounter ID** from `data/synthea_csv/encounters.csv` (the `Id` column)
3. Type a question in the search box
4. Click **Get answers**

> Both Patient ID and Encounter ID are required. They look like UUIDs, for example:
> `45dff467-def6-2132-9e5c-0a836d754d92`

---

## Example queries

| Query | What it returns |
|-------|----------------|
| `what is the patient name` | Full name from demographics |
| `when was the patient born` | Date of birth |
| `what is the patient sex` | Gender |
| `what are the diagnoses` | List of conditions |
| `what medications is the patient taking` | Medication list |
| `list patient procedures` | Procedures performed |
| `what is the sodium level` | Lab value for sodium |
| `is simvastatin prescribed` | Yes/No medication check |
| `what is the visit reason` | Reason for the encounter |
| `what is the discharge plan` | Follow-up plan |

---

## Rebuild from scratch (optional)

Only needed if you want to use your own Synthea data or retrain the model:

```bash
# 1. Generate clinical notes from your Synthea CSVs
python artifacts/make_notes_from_csv.py

# 2. Fine-tune the SBERT bi-encoder on those notes
python artifacts/finetune_sbert.py

# 3. Build the search index from the fine-tuned model
python build_index.py

# 4. Launch the app
streamlit run app.py
```

---

## Evaluate extraction accuracy

```bash
python evaluate.py
# or test on more samples:
python evaluate.py --n_samples 200
```

This filters by patient ID + encounter ID (exactly like the app) and checks whether the rule-based extractor pulls the correct answer from the note. Reports `Extract%` per intent (name, birthdate, diagnoses, medications, etc.).

---

## How it works

1. **Note generation** — Synthea CSVs (patients, encounters, conditions, medications, observations, procedures) are merged into structured discharge-style notes, one note per encounter.
2. **Fine-tuning** — Bio_ClinicalBERT is fine-tuned as a SimCSE bi-encoder: the same note is passed through the model twice with different dropout masks, forming a positive pair. All other notes in the batch are negatives.
3. **Indexing** — Every note is embedded with the fine-tuned encoder and L2-normalized. Embeddings are saved to `artifacts/search_index/embeddings.npy`.
4. **Retrieval** — At query time, the app filters by patient ID + encounter ID (leaving exactly 1 candidate note), embeds the query, and computes cosine similarity.
5. **Extraction** — Intent-based rules pull the specific answer (name, birthdate, diagnoses, etc.) out of the matched note.

---

## Model details

| Property | Value |
|----------|-------|
| Base model | `emilyalsentzer/Bio_ClinicalBERT` |
| Fine-tuning method | SimCSE (unsupervised contrastive learning) |
| Embedding dim | 384 (projected from 768) |
| Max sequence length | 256 tokens |
| Training data | Synthea synthetic EHR notes |
| Epochs | 5 |
| Batch size | 16 (effective 64 with gradient accumulation) |
