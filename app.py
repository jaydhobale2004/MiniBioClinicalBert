import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import streamlit as st
import torch
from rank_bm25 import BM25Okapi
from transformers import AutoModel, AutoTokenizer


MODEL_DIR = "artifacts/sbert_bi_encoder"
INDEX_DIR = Path("artifacts/search_index")
MAX_LEN = 256
BM25_WEIGHT = 0.3


def get_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def infer_drug_from_query(q: str) -> Optional[str]:
    q = q.strip().lower()
    m = re.search(r"\bis\s+(.+?)\s+prescribed\b", q)
    if m:
        return m.group(1).strip(" ?\"'")
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
    out = Path(MODEL_DIR)
    has_model = out.exists() and any(
        f.suffix in (".json", ".bin", ".safetensors") for f in out.iterdir()
    )
    if not has_model:
        model_path = "emilyalsentzer/Bio_ClinicalBERT"
        st.warning(
            f"SBERT model not found at '{MODEL_DIR}'. "
            f"Using {model_path} as fallback. "
            "Run `python artifacts/finetune_sbert.py` then `python build_index.py` to activate SBERT."
        )
    else:
        model_path = MODEL_DIR
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).to(device)
    model.eval()
    return tok, model, device


@st.cache_resource
def load_index():
    emb_path = INDEX_DIR / "embeddings.npy"
    meta_path = INDEX_DIR / "meta.jsonl"
    if not emb_path.exists() or not meta_path.exists():
        raise FileNotFoundError("Index not found. Run: python build_index.py")

    embs = np.load(emb_path)  # (N, dim) already L2-normalized
    meta = []
    with meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            meta.append(json.loads(line))

    tokenized_corpus = [doc["text"].lower().split() for doc in meta]
    bm25 = BM25Okapi(tokenized_corpus)
    return embs, meta, bm25


def embed_query(query: str, tok, model, device) -> np.ndarray:
    encoded = tok(
        query,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LEN,
        padding=True,
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        out = model(**encoded)
        vec = mean_pool(out.last_hidden_state, encoded["attention_mask"])

    vec = vec.cpu().numpy()[0].astype(np.float32)
    return normalize(vec)


def hybrid_search(
    query: str,
    query_vec: np.ndarray,
    embs: np.ndarray,
    bm25: BM25Okapi,
    candidate_idx: List[int],
    k: int,
    bm25_weight: float = BM25_WEIGHT,
) -> Tuple[np.ndarray, np.ndarray]:
    sub_embs = embs[candidate_idx]

    dense_scores = sub_embs @ query_vec
    dense_min, dense_max = dense_scores.min(), dense_scores.max()
    dense_range = dense_max - dense_min
    # when there's only 1 candidate (or all score identically), min-max collapses to 0
    # use raw cosine similarity shifted to [0, 1] instead
    if dense_range < 1e-6:
        dense_norm = (dense_scores + 1) / 2
    else:
        dense_norm = (dense_scores - dense_min) / dense_range

    query_tokens = query.lower().split()
    all_bm25 = np.array(bm25.get_scores(query_tokens))
    bm25_scores = all_bm25[candidate_idx]
    bm25_max = bm25_scores.max()
    bm25_norm = bm25_scores / (bm25_max + 1e-9)

    hybrid = (1 - bm25_weight) * dense_norm + bm25_weight * bm25_norm

    k = min(k, len(hybrid))
    idx_sub = np.argpartition(-hybrid, kth=k - 1)[:k]
    idx_sub = idx_sub[np.argsort(-hybrid[idx_sub])]
    return idx_sub, hybrid[idx_sub]


def detect_intent(q: str) -> str:
    q = q.lower().strip()

    if any(w in q for w in ["name", "who is", "what is the name", "patient name"]):
        return "name"
    if any(w in q for w in ["marital", "married", "marital status"]):
        return "marital_status"
    if any(w in q for w in ["birthdate", "date of birth", "dob", "born"]):
        return "birthdate"
    if any(w in q for w in ["encounter id", "visit id"]):
        return "encounter_id"
    if any(w in q for w in ["patient id"]):
        return "patient_id"
    if any(w in q for w in ["date", "dates", "when was the visit", "visit date"]):
        return "encounter_dates"
    if any(w in q for w in ["encounter class", "visit class", "class of encounter"]):
        return "encounter_class"
    if any(w in q for w in ["visit reason", "reason for visit", "why was the patient seen", "reason description"]):
        return "visit_reason"
    if any(w in q for w in ["reason code", "code for visit reason"]):
        return "reason_code"
    if any(w in q for w in ["sex", "gender"]):
        return "sex"
    if "race" in q:
        return "race"
    if "ethnicity" in q:
        return "ethnicity"
    if any(w in q for w in ["diagnosis", "diagnoses", "condition", "conditions", "problem"]):
        return "diagnoses"
    if any(w in q for w in ["procedure", "procedures", "operation", "operations", "surgery", "surgeries"]):
        return "procedures"
    if "prescribed" in q and infer_drug_from_query(q):
        return "med_check"
    if any(w in q for w in ["medication", "medications", "medicine", "medicines", "med", "meds", "drug", "drugs", "prescribed"]):
        return "medications"
    if any(w in q for w in ["plan", "follow-up", "follow up", "next steps", "discharge plan"]):
        return "plan"

    lab_terms = [
        "lab", "labs", "observation", "observations",
        "sodium", "potassium", "glucose", "creatinine", "urea",
        "chloride", "calcium", "bilirubin", "albumin", "protein",
        "alt", "ast", "alkaline", "cholesterol", "triglyceride", "hdl", "ldl",
    ]
    if any(w in q for w in lab_terms):
        return "labs"

    return "general"


def infer_lab_keyword(query: str) -> Optional[str]:
    q = query.lower()
    candidates = [
        "sodium", "potassium", "glucose", "creatinine", "urea",
        "chloride", "calcium", "bilirubin", "albumin", "protein",
        "alanine", "aspartate", "alkaline", "cholesterol", "triglyceride", "hdl", "ldl",
        "carbon dioxide",
    ]
    for c in candidates:
        if c in q:
            return c
    return None


def parse_observation_line(line: str) -> Optional[Dict[str, str]]:
    parts = [p.strip() for p in line.split("|") if p.strip()]
    if not parts:
        return None

    parsed = {"name": "", "value": "", "date": ""}
    first = parts[0]
    if ":" in first:
        name, val = first.split(":", 1)
        parsed["name"] = name.strip()
        parsed["value"] = val.strip()
    else:
        parsed["name"] = first.strip()

    for part in parts[1:]:
        if ":" not in part:
            # bare value with no key — treat it as the date
            if not parsed["date"]:
                parsed["date"] = part.strip()
            continue
        key, val = part.split(":", 1)
        parsed[key.strip().lower()] = val.strip()

    return parsed


def extract_labs(text: str) -> list[dict]:
    m = re.search(r"LABS / OBSERVATIONS:\n(.*?)(\n\n|$)", text, flags=re.S)
    if not m:
        return []

    out = []
    for ln in m.group(1).splitlines():
        ln = ln.strip()
        if not ln.startswith("-"):
            continue
        ln = ln.lstrip("-").strip()
        parsed = parse_observation_line(ln)
        if not parsed or not parsed["name"]:
            continue
        if parsed.get("date"):
            parsed["date"] = parsed["date"]
        out.append(parsed)
    return out


def extract_lab_matches(note_texts: list[str], keyword: str) -> list[dict]:
    kw = keyword.lower()
    matches = []
    for t in note_texts:
        for lab in extract_labs(t):
            if kw in lab["name"].lower() or kw in lab["value"].lower():
                matches.append(lab)
    return matches


def extract_observation_lines(note_texts: list[str]) -> list[str]:
    seen = set()
    out = []
    for text in note_texts:
        for item in extract_bullets_from_section(text, "LABS / OBSERVATIONS:"):
            if item not in seen:
                seen.add(item)
                out.append(item)
    return out


def extract_bullets_from_section(text: str, header: str) -> List[str]:
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


def extract_demographic_value(text: str, key: str) -> Optional[str]:
    for ln in extract_bullets_from_section(text, "DEMOGRAPHICS:"):
        if ln.lower().startswith(key.lower() + ":"):
            return ln.split(":", 1)[1].strip()
    return None


def extract_header_value(text: str, key: str) -> Optional[str]:
    prefix = key + ":"
    for ln in text.splitlines():
        if ln.startswith(prefix):
            return ln.split(":", 1)[1].strip()
    return None


def extract_section_lines(text: str, header: str) -> List[str]:
    return extract_bullets_from_section(text, header)


def med_contains(med_line: str, drug: str) -> bool:
    return drug.lower() in med_line.lower()


def render_answer(query: str, intent: str, retrieved: List[Dict], top_k: int):
    st.subheader("Answer (extracted)")

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
            st.info("Tip: increase Top-K or filter by patient_id/encounter_id.")
        return

    if intent in ["name", "birthdate", "sex", "race", "ethnicity", "marital_status"]:
        key_map = {
            "name": "Name",
            "birthdate": "Birthdate",
            "sex": "Sex",
            "race": "Race",
            "ethnicity": "Ethnicity",
            "marital_status": "Marital Status",
        }
        key = key_map[intent]
        found = []
        for r in retrieved:
            val = extract_demographic_value(r["text"], key)
            if val:
                found.append((r.get("patient_id"), r.get("encounter_id"), val))
        if not found:
            st.write(f"No {key} found in DEMOGRAPHICS section of the top results.")
            st.info("Tip: Use the sidebar patient_id filter to get one specific patient.")
            return
        st.write(f"Found {len(found)} matches (search may include multiple patients).")
        for pid, eid, val in found[:top_k]:
            st.write(f"- patient={pid} | encounter={eid} | {key}={val}")
        return

    if intent in ["encounter_id", "patient_id", "encounter_dates", "encounter_class", "visit_reason", "reason_code"]:
        key_map = {
            "encounter_id": "Encounter ID",
            "patient_id": "Patient ID",
            "encounter_dates": "Dates",
            "encounter_class": "Encounter Class",
            "visit_reason": "Visit Reason",
            "reason_code": "Reason Description",
        }
        key = key_map[intent]
        found = []
        for r in retrieved:
            val = extract_header_value(r["text"], key)
            if val:
                if intent == "reason_code" and "|" in val:
                    val = val.split("|", 1)[1].strip()
                found.append((r.get("patient_id"), r.get("encounter_id"), val))
        if not found:
            st.write(f"No {key} found in the top results.")
            return
        st.write(f"Found {len(found)} matches.")
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
        seen, uniq = set(), []
        for d in all_diags:
            if d not in seen:
                seen.add(d)
                uniq.append(d)
        st.write("Likely Diagnoses:")
        for d in uniq[:10]:
            st.write(f"- {d}")
        return

    if intent == "procedures":
        all_procs = []
        for r in retrieved:
            all_procs += extract_bullets_from_section(r["text"], "PROCEDURES:")
        if not all_procs:
            st.write("No procedures section found in the top results.")
            return
        seen, uniq = set(), []
        for p in all_procs:
            if p not in seen:
                seen.add(p)
                uniq.append(p)
        st.write("Likely Procedures:")
        for p in uniq[:10]:
            st.write(f"- {p}")
        return

    if intent == "medications":
        all_meds = []
        for r in retrieved:
            all_meds += extract_bullets_from_section(r["text"], "MEDICATIONS:")
        if not all_meds:
            st.write("No medications section found in the top results.")
            return
        seen, uniq = set(), []
        for m in all_meds:
            if m not in seen:
                seen.add(m)
                uniq.append(m)
        st.write("Likely Medications:")
        for m in uniq[:10]:
            st.write(f"- {m}")
        return

    if intent == "labs":
        kw = infer_lab_keyword(query)
        if not kw:
            obs = extract_observation_lines([r["text"] for r in retrieved])
            if not obs:
                st.write("No labs or observations section found in the top results.")
                return
            st.write("Likely Labs / Observations:")
            for item in obs[:10]:
                st.write(f"- {item}")
            return
        hits = extract_lab_matches([r["text"] for r in retrieved], kw)
        if not hits:
            st.write(f"No '{kw}' found in LABS / OBSERVATIONS in the retrieved notes.")
            st.info("Tip: Use patient_id or encounter_id filter to avoid mixing patients.")
            return
        st.write(f"Found {len(hits)} '{kw}' values:")
        seen, shown = set(), 0
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

    if intent == "plan":
        all_plan = []
        for r in retrieved:
            all_plan += extract_section_lines(r["text"], "PLAN:")
        if not all_plan:
            st.write("No plan section found in the top results.")
            return
        seen, uniq = set(), []
        for item in all_plan:
            if item not in seen:
                seen.add(item)
                uniq.append(item)
        st.write("Likely Plan:")
        for item in uniq[:10]:
            st.write(f"- {item}")
        return

    st.write("Try asking about: birthdate / marital status / encounter dates / visit reason / reason code / diagnoses / procedures / medications / labs / plan.")


def filter_retrieved_by_intent(intent: str, retrieved: List[Dict]) -> List[Dict]:
    header_map = {
        "diagnoses": "DIAGNOSES / CONDITIONS:",
        "procedures": "PROCEDURES:",
        "medications": "MEDICATIONS:",
        "med_check": "MEDICATIONS:",
        "labs": "LABS / OBSERVATIONS:",
    }
    header = header_map.get(intent)
    if not header:
        return retrieved
    filtered = [r for r in retrieved if header in r.get("text", "")]
    return filtered or retrieved


def filter_candidate_indices_by_intent(intent: str, meta: List[Dict], candidate_idx: List[int]) -> List[int]:
    header_map = {
        "diagnoses": "DIAGNOSES / CONDITIONS:",
        "procedures": "PROCEDURES:",
        "medications": "MEDICATIONS:",
        "med_check": "MEDICATIONS:",
        "labs": "LABS / OBSERVATIONS:",
        "plan": "PLAN:",
    }
    header = header_map.get(intent)
    if not header:
        return candidate_idx
    filtered = [i for i in candidate_idx if header in meta[i].get("text", "")]
    return filtered or candidate_idx


def main():
    st.title("MiniBioBERT Clinical Search UI")

    tok, model, device = load_encoder()
    embs, meta, bm25 = load_index()

    st.write(f"Running on **{device}**. Model: **SBERT bi-encoder** | Notes indexed: **{len(meta)}**")

    st.sidebar.header("Search Settings")
    top_k = st.sidebar.slider("Top-K results", 1, 20, 5)
    bm25_w = st.sidebar.slider("BM25 weight (0=pure dense, 1=pure BM25)", 0.0, 1.0, BM25_WEIGHT, step=0.05)

    patient_filter = st.sidebar.text_input("Filter by patient_id (required)", value="").strip()
    encounter_filter = st.sidebar.text_input("Filter by encounter_id (required)", value="").strip()

    if not patient_filter:
        st.error("Patient ID is required. Please enter a patient_id in the sidebar.")
        return

    if not encounter_filter:
        st.error("Encounter ID is required. Please enter an encounter_id in the sidebar.")
        return

    candidate_idx = list(range(len(meta)))
    candidate_idx = [i for i in candidate_idx if meta[i].get("patient_id") == patient_filter and meta[i].get("encounter_id") == encounter_filter]

    query = st.text_input("Ask a question (semantic search)", value="birthdate")

    if st.button("Get answers"):
        if len(candidate_idx) == 0:
            st.error("No notes match the specified patient_id and encounter_id. Please check the IDs.")
            return

        intent = detect_intent(query)
        candidate_idx = filter_candidate_indices_by_intent(intent, meta, candidate_idx)
        qv = embed_query(query, tok, model, device)

        idx_sub, scores = hybrid_search(query, qv, embs, bm25, candidate_idx, top_k, bm25_w)
        idx = [candidate_idx[i] for i in idx_sub]
        retrieved = [meta[i] for i in idx]
        retrieved = filter_retrieved_by_intent(intent, retrieved)

        render_answer(query, intent, retrieved, top_k)

        st.subheader("Top retrieved notes (evidence)")
        for rank, (i, score) in enumerate(zip(idx, scores), start=1):
            r = meta[i]
            title = f"{rank}) score={float(score):.4f} | patient={r.get('patient_id')} | encounter={r.get('encounter_id')}"
            with st.expander(title):
                st.text(r["text"])

        if intent == "birthdate" and not patient_filter:
            st.info("For a single correct birthdate: copy a patient_id from evidence and paste it into the sidebar filter.")


if __name__ == "__main__":
    main()
