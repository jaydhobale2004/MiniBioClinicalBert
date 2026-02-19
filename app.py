"""
app.py - MiniBioBERT Clinical Search UI

How it works:
1) Load embeddings.npy and meta.jsonl
2) Embed user's query with the same MiniBioBERT encoder
3) Find the most similar notes via cosine similarity
4) Extract answer from the retrieved notes using simple rules

Important:
- This is NOT a chat model. It does retrieval + extraction.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import streamlit as st
import torch
from transformers import AutoModel, AutoTokenizer


# -------------------------
# Configuration
# -------------------------
MODEL_DIR = Path("artifacts/mini_biobert_mlm")
INDEX_DIR = Path("artifacts/search_index")
MAX_LEN = 256


def get_device() -> torch.device:
    """GPU if available else CPU."""
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import re

def infer_drug_from_query(q: str) -> str | None:
    q = q.strip().lower()

    # patterns like: "is simvastatin prescribed?" / "is simvastatin prescribed"
    m = re.search(r"\bis\s+(.+?)\s+prescribed\b", q)
    if m:
        return m.group(1).strip(" ?\"'")

    # patterns like: "does the patient take simvastatin"
    m = re.search(r"\btake\s+(.+?)\b", q)
    if m:
        return m.group(1).strip(" ?\"'")

    return None

def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def normalize(vec: np.ndarray) -> np.ndarray:
    return vec / (np.linalg.norm(vec) + 1e-9)


@st.cache_resource
def load_encoder():
    device = get_device()
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModel.from_pretrained(MODEL_DIR).to(device)
    model.eval()
    return tok, model, device


@st.cache_resource
def load_index():
    emb_path = INDEX_DIR / "embeddings.npy"
    meta_path = INDEX_DIR / "meta.jsonl"
    if not emb_path.exists() or not meta_path.exists():
        raise FileNotFoundError("Index not found. Run: python build_index.py")

    embs = np.load(emb_path)  # (N, dim) already normalized
    meta = []
    with meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            meta.append(json.loads(line))
    return embs, meta


def embed_query(query: str, tok, model, device) -> np.ndarray:
    """Encode query to a normalized embedding vector."""
    encoded = tok(
        query,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LEN,
        padding=True
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        out = model(**encoded)
        vec = mean_pool(out.last_hidden_state, encoded["attention_mask"])  # (1, hidden)

    vec = vec.cpu().numpy()[0].astype(np.float32)
    return normalize(vec)


def top_k_search(query_vec: np.ndarray, embs: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cosine similarity because embeddings are normalized:
      sim = dot(note_vec, query_vec)
    """
    scores = embs @ query_vec  # (N,)
    k = min(k, len(scores))
    idx = np.argpartition(-scores, kth=k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return idx, scores[idx]


# -------------------------
# Answer extraction (intent-based)
# -------------------------

def detect_intent(q: str) -> str:
    q = q.lower().strip()

    if any(w in q for w in ["birthdate", "date of birth", "dob", "born"]):
        return "birthdate"
    if any(w in q for w in ["sex", "gender"]):
        return "sex"
    if "race" in q:
        return "race"
    if "ethnicity" in q:
        return "ethnicity"

    if any(w in q for w in ["diagnosis", "diagnoses", "condition", "conditions", "problem"]):
        return "diagnoses"
    if "prescribed" in q and infer_drug_from_query(q):
        return "med_check"
    if any(w in q for w in ["medication", "medications", "drug", "drugs", "prescribed"]):
        return "medications"

    # ✅ LABS / OBSERVATIONS intent
    lab_terms = [
        "lab", "labs", "observation", "observations",
        "sodium", "potassium", "glucose", "creatinine", "urea",
        "chloride", "calcium", "bilirubin", "albumin", "protein",
        "alt", "ast", "alkaline", "cholesterol", "triglyceride", "hdl", "ldl"
    ]
    if any(w in q for w in lab_terms):
        return "labs"

    return "general"

import re

def infer_lab_keyword(query: str) -> str | None:
    """
    Infer which lab the user is asking about.
    For now, we match simple common analytes.
    """
    q = query.lower()
    candidates = [
        "sodium", "potassium", "glucose", "creatinine", "urea",
        "chloride", "calcium", "bilirubin", "albumin", "protein",
        "alanine", "aspartate", "alkaline", "cholesterol", "triglyceride", "hdl", "ldl",
        "carbon dioxide"
    ]
    for c in candidates:
        if c in q:
            return c
    return None


def extract_labs(text: str) -> list[dict]:
    """
    Extract all LABS / OBSERVATIONS bullet lines.
    Each line in your notes looks like:
      - Sodium [Moles/volume] in Blood: 138.1 mmol/L | 2013-...
    """
    m = re.search(r"LABS / OBSERVATIONS:\n(.*?)(\n\n|$)", text, flags=re.S)
    if not m:
        return []

    out = []
    for ln in m.group(1).splitlines():
        ln = ln.strip()
        if not ln.startswith("-"):
            continue
        ln = ln.lstrip("-").strip()

        # Split date (optional)
        parts = [p.strip() for p in ln.split("|")]
        main = parts[0]
        date = parts[1] if len(parts) > 1 else ""

        # Split name vs value
        # main: "<name>: <value units>"
        if ":" not in main:
            continue
        name, val = main.split(":", 1)
        out.append({"name": name.strip(), "value": val.strip(), "date": date})

    return out


def extract_lab_matches(note_texts: list[str], keyword: str) -> list[dict]:
    """
    Find all lab entries that mention the keyword (case-insensitive).
    """
    kw = keyword.lower()
    matches = []
    for t in note_texts:
        labs = extract_labs(t)
        for lab in labs:
            if kw in lab["name"].lower():
                matches.append(lab)
    return matches


def extract_bullets_from_section(text: str, header: str) -> List[str]:
    """
    Extract bullet lines under "HEADER:" until blank line.
    Example headers in your notes:
      DEMOGRAPHICS:
      DIAGNOSES / CONDITIONS:
      MEDICATIONS:
    """
    pattern = rf"{re.escape(header)}\n(.*?)(\n\n|$)"
    m = re.search(pattern, text, flags=re.S)
    if not m:
        return []

    bullets = []
    for ln in m.group(1).splitlines():
        ln = ln.strip()
        if ln.startswith("-"):
            bullets.append(ln.lstrip("-").strip())
    return bullets


def extract_demographic_value(text: str, key: str) -> str | None:
    """
    From DEMOGRAPHICS bullets like:
      Sex: F
      Race: white
      Ethnicity: nonhispanic
      Birthdate: 1990-01-01
    """
    demo = extract_bullets_from_section(text, "DEMOGRAPHICS:")
    for ln in demo:
        if ln.lower().startswith(key.lower() + ":"):
            return ln.split(":", 1)[1].strip()
    return None

def med_contains(med_line: str, drug: str) -> bool:
    return drug.lower() in med_line.lower()


def render_answer(query: str,intent: str, retrieved: List[Dict], top_k: int):
    """
    Display extracted answer based on intent.
    retrieved contains dicts with {patient_id, encounter_id, text}.
    """
    st.subheader("Answer (extracted)")
   
    # ✅ add this first
    if intent == "med_check":
        drug = infer_drug_from_query(query)
        if not drug:
            st.write("Could not detect the drug name. Try: 'is simvastatin prescribed?'")
            return

        all_meds = []
        for r in retrieved:
            all_meds += extract_bullets_from_section(r["text"], "MEDICATIONS:")

        hits = [m for m in all_meds if med_contains(m, drug)]

        if hits:
            st.write(f"**YES** — '{drug}' is prescribed in the retrieved notes.")
            for h in hits[:5]:
                st.write(f"- {h}")
        else:
            st.write(f"**NO** — '{drug}' was not found in the retrieved MEDICATIONS sections.")
            st.info("Tip: increase Top-K or filter by patient_id/encounter_id to avoid mixing patients.")
        return

   
    if intent in ["birthdate", "sex", "race", "ethnicity"]:
        key_map = {
            "birthdate": "Birthdate",
            "sex": "Sex",
            "race": "Race",
            "ethnicity": "Ethnicity"
        }
        key = key_map[intent]

        found = []
        for r in retrieved:
            val = extract_demographic_value(r["text"], key)
            if val:
                found.append((r.get("patient_id"), r.get("encounter_id"), val))

        if not found:
            st.write(f"No {key} found in DEMOGRAPHICS section of the top results.")
            st.info("Tip: Use the sidebar patient_id filter to get one specific patient’s birthdate.")
            return

        st.write(f"Found {len(found)} matches (because search may include multiple patients).")
        for pid, eid, val in found[:top_k]:
            st.write(f"- patient={pid} | encounter={eid} | {key}={val}")
        return

    if intent == "diagnoses":
        all_diags = []
        for r in retrieved:
            all_diags += extract_bullets_from_section(r["text"], "DIAGNOSES / CONDITIONS:")

        if not all_diags:
            st.write("No diagnoses section found in the top results.")
            return

        # de-duplicate while preserving order
        seen = set()
        uniq = []
        for d in all_diags:
            if d not in seen:
                seen.add(d)
                uniq.append(d)

        st.write("Likely Diagnoses:")
        for d in uniq[:10]:
            st.write(f"- {d}")
        return

    if intent == "medications":
        all_meds = []
        for r in retrieved:
            all_meds += extract_bullets_from_section(r["text"], "MEDICATIONS:")

        if not all_meds:
            st.write("No medications section found in the top results.")
            return

        seen = set()
        uniq = []
        for m in all_meds:
            if m not in seen:
                seen.add(m)
                uniq.append(m)

        st.write("Likely Medications:")
        for m in uniq[:10]:
            st.write(f"- {m}")
        return
    elif intent == "labs":
        kw = infer_lab_keyword(query)  # query must be accessible here; if not, pass it into render_answer
        if not kw:
            st.write("Ask a specific lab name (e.g., sodium, glucose, creatinine).")
            return

        note_texts = [r["text"] for r in retrieved]
        hits = extract_lab_matches(note_texts, kw)

        st.subheader("Answer (extracted)")
        if not hits:
            st.write(f"No '{kw}' found in LABS / OBSERVATIONS in the retrieved notes.")
            st.info("Tip: Use patient_id or encounter_id filter to avoid mixing patients.")
            return

        st.write(f"Found {len(hits)} '{kw}' values:")
    # de-dupe
        seen = set()
        shown = 0
        for h in hits:
            key = (h["name"], h["value"], h["date"])
            if key in seen:
                continue
            seen.add(key)
            st.write(f"- {h['name']}: {h['value']}" + (f" | {h['date']}" if h["date"] else ""))
            shown += 1
            if shown >= 10:
                break
    return


    st.write("Try asking about: birthdate / sex / race / ethnicity / diagnoses / medications.")


def main():
    st.title("MiniBioBERT Clinical Search UI")

    tok, model, device = load_encoder()
    embs, meta = load_index()

    st.write(f"Running on **{device}**.")

    # -------------------------
    # Sidebar filters (important for correct birthdate)
    # -------------------------
    st.sidebar.header("Search Settings")

    top_k = st.sidebar.slider("Top-K results", 1, 20, 5)

    patient_filter = st.sidebar.text_input("Filter by patient_id (optional)", value="").strip()
    encounter_filter = st.sidebar.text_input("Filter by encounter_id (optional)", value="").strip()

    # Build candidate index list based on filters
    candidate_idx = list(range(len(meta)))
    if patient_filter:
        candidate_idx = [i for i in candidate_idx if meta[i].get("patient_id") == patient_filter]
    if encounter_filter:
        candidate_idx = [i for i in candidate_idx if meta[i].get("encounter_id") == encounter_filter]

    # Subset embeddings if filtered
    embs_sub = embs[candidate_idx] if candidate_idx else np.empty((0, embs.shape[1]), dtype=np.float32)

    # -------------------------
    # Query input
    # -------------------------
    query = st.text_input("Ask a question (semantic search)", value="birthdate")

    if st.button("Get answers"):
        if len(candidate_idx) == 0:
            st.error("No notes match your filters. Clear patient_id/encounter_id filter and try again.")
            return

        intent = detect_intent(query)

        qv = embed_query(query, tok, model, device)
        idx_sub, scores = top_k_search(qv, embs_sub, top_k)

        # map subset indices back to original
        idx = [candidate_idx[i] for i in idx_sub]
        retrieved = [meta[i] for i in idx]

        # Render extracted answer
        render_answer(query, intent, retrieved, top_k)

        # Show evidence notes
        st.subheader("Top retrieved notes (evidence)")
        for rank, (i, score) in enumerate(zip(idx, scores), start=1):
            r = meta[i]
            title = f"{rank}) score={float(score):.4f} | patient={r.get('patient_id')} | encounter={r.get('encounter_id')}"
            with st.expander(title):
                st.text(r["text"])

        # Helpful tip for your use case
        if intent == "birthdate" and not patient_filter:
            st.info("For a single correct birthdate: copy a patient_id from evidence and paste it into the sidebar filter.")


if __name__ == "__main__":
    main()
