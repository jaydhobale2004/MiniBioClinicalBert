"""
evaluate.py

Simulates the app exactly: for each test case, patient_id + encounter_id
are used to pin down the single matching note, then we ask intent-based
questions and check whether the extractor surfaces the right answer.

The only meaningful metric here is Extraction% — retrieval is always
100% because there is exactly 1 candidate note after filtering.

Usage:
    python evaluate.py
    python evaluate.py --n_samples 200
"""

import argparse
import json
import re
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

MODEL_DIR      = "artifacts/sbert_bi_encoder"
INDEX_DIR      = Path("artifacts/search_index")
MAX_LEN        = 256
FALLBACK_MODEL = "emilyalsentzer/Bio_ClinicalBERT"

INTENT_QUERIES = {
    "name":        ["what is the patient name", "who is the patient", "patient name"],
    "birthdate":   ["what is the date of birth", "when was the patient born", "patient birthdate"],
    "sex":         ["what is the patient sex", "what is the gender of the patient"],
    "race":        ["what is the patient race"],
    "ethnicity":   ["what is the patient ethnicity"],
    "diagnoses":   ["what are the diagnoses", "list patient conditions", "what conditions does the patient have"],
    "medications": ["what medications is the patient taking", "list patient medications"],
}

DEMO_KEY_MAP = {
    "name": "Name", "birthdate": "Birthdate",
    "sex": "Sex", "race": "Race", "ethnicity": "Ethnicity",
}


# ── Encoder ───────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def mean_pool(last_hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.unsqueeze(-1).float()
    return (last_hidden * m).sum(1) / m.sum(1).clamp(min=1e-9)


def normalize(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-9)


def load_encoder():
    device = get_device()
    out = Path(MODEL_DIR)
    has_model = out.exists() and any(f.suffix in (".json", ".bin", ".safetensors") for f in out.iterdir())
    model_path = MODEL_DIR if has_model else FALLBACK_MODEL
    if not has_model:
        print(f"[WARN] Fine-tuned model not found — using {FALLBACK_MODEL}")
    tok   = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).to(device)
    model.eval()
    print(f"Encoder: {model_path}  |  device: {device}")
    return tok, model, device


def embed(text: str, tok, model, device) -> np.ndarray:
    enc = tok(text, return_tensors="pt", truncation=True, max_length=MAX_LEN, padding=True)
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = model(**enc)
        vec = mean_pool(out.last_hidden_state, enc["attention_mask"])
    return normalize(vec.cpu().numpy()[0].astype(np.float32))


# ── Extraction helpers ────────────────────────────────────────────────────────

def extract_section_bullets(text: str, header: str) -> List[str]:
    m = re.search(rf"{re.escape(header)}\n(.*?)(\n\n|$)", text, flags=re.S)
    if not m:
        return []
    return [ln.lstrip("-").strip() for ln in m.group(1).splitlines() if ln.strip().startswith("-")]


def extract_demo_value(text: str, key: str) -> Optional[str]:
    for ln in extract_section_bullets(text, "DEMOGRAPHICS:"):
        if ln.lower().startswith(key.lower() + ":"):
            return ln.split(":", 1)[1].strip()
    return None


def extract_answer(note_text: str, intent: str) -> Optional[str]:
    if intent in DEMO_KEY_MAP:
        return extract_demo_value(note_text, DEMO_KEY_MAP[intent])
    if intent == "diagnoses":
        items = extract_section_bullets(note_text, "DIAGNOSES / CONDITIONS:")
        return items[0] if items else None
    if intent == "medications":
        items = extract_section_bullets(note_text, "MEDICATIONS:")
        return items[0] if items else None
    return None


# ── Test-case generation ──────────────────────────────────────────────────────

def build_test_cases(meta: List[Dict], n_samples: int, seed: int = 42) -> List[Dict]:
    random.seed(seed)
    sampled = random.sample(meta, min(n_samples, len(meta)))
    cases = []

    for note in sampled:
        for intent, queries in INTENT_QUERIES.items():
            gt = extract_answer(note["text"], intent)
            if gt is None:
                continue
            cases.append({
                "patient_id":   note["patient_id"],
                "encounter_id": note["encounter_id"],
                "intent":       intent,
                "query":        random.choice(queries),
                "ground_truth": gt,
                "note_text":    note["text"],
            })

    return cases


# ── Evaluation ────────────────────────────────────────────────────────────────

def run_evaluation(
    cases: List[Dict],
    embs: np.ndarray,
    meta: List[Dict],
    tok, model, device,
) -> Tuple[List[Dict], List[Dict]]:
    eid_to_idx = {n["encounter_id"]: i for i, n in enumerate(meta)}

    results = []
    errors  = []

    print(f"\nEvaluating {len(cases)} cases  (patient_id + encounter_id filter = 1 note each)...")

    for i, case in enumerate(cases):
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(cases)}")

        note_idx  = eid_to_idx[case["encounter_id"]]
        note_text = meta[note_idx]["text"]

        # cosine similarity: how well does the query match the note?
        qv   = embed(case["query"],   tok, model, device)
        nv   = embs[note_idx]
        score = float(np.dot(qv, nv))

        # extraction: does the extractor pull the right answer from this note?
        extracted = extract_answer(note_text, case["intent"])
        gt_lower  = case["ground_truth"].lower()
        correct   = extracted is not None and gt_lower in extracted.lower()

        results.append({
            **case,
            "extracted": extracted,
            "correct":   correct,
            "score":     score,
        })

        if not correct:
            errors.append({
                "patient_id":   case["patient_id"],
                "encounter_id": case["encounter_id"],
                "intent":       case["intent"],
                "query":        case["query"],
                "ground_truth": case["ground_truth"],
                "extracted":    extracted,
                "score":        score,
            })

    return results, errors


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_metrics(results: List[Dict]):
    intents = sorted({r["intent"] for r in results})

    print("\n" + "=" * 55)
    print("EXTRACTION ACCURACY  (patient_id + encounter_id filter)")
    print("=" * 55)
    print(f"{'Intent':<16}  {'N':>5}  {'Correct':>8}  {'Extract%':>9}  {'Avg Score':>10}")
    print("-" * 55)

    overall_correct = 0
    overall_total   = 0

    for intent in intents:
        rows    = [r for r in results if r["intent"] == intent]
        n       = len(rows)
        correct = sum(r["correct"] for r in rows)
        pct     = 100 * correct / n if n else 0
        avg_score = np.mean([r["score"] for r in rows])
        print(f"{intent:<16}  {n:>5}  {correct:>8}  {pct:>8.1f}%  {avg_score:>10.4f}")
        overall_correct += correct
        overall_total   += n

    overall_pct = 100 * overall_correct / overall_total if overall_total else 0
    overall_score = np.mean([r["score"] for r in results])
    print("-" * 55)
    print(f"{'OVERALL':<16}  {overall_total:>5}  {overall_correct:>8}  {overall_pct:>8.1f}%  {overall_score:>10.4f}")
    print("=" * 55)
    print("\nExtract% = extractor found the right answer inside the pinned note.")
    print("Avg Score = cosine similarity of the query against that note.\n")


def print_errors(errors: List[Dict], max_show: int = 10):
    by_intent = defaultdict(int)
    for e in errors:
        by_intent[e["intent"]] += 1

    print("=" * 55)
    print(f"EXTRACTION FAILURES  ({len(errors)} total)")
    print("=" * 55)
    for intent, cnt in sorted(by_intent.items(), key=lambda x: -x[1]):
        print(f"  {intent:<16}  {cnt}")

    print(f"\n-- First {min(max_show, len(errors))} failures --")
    for e in errors[:max_show]:
        print(f"\n  patient_id  : {e['patient_id']}")
        print(f"  encounter_id: {e['encounter_id']}")
        print(f"  intent      : {e['intent']}")
        print(f"  query       : {e['query']}")
        print(f"  expected    : {e['ground_truth']}")
        print(f"  extracted   : {e['extracted']}")
        print(f"  score       : {e['score']:.4f}")


def save_errors(errors: List[Dict], path: str = "eval_errors.json"):
    Path(path).write_text(json.dumps(errors, indent=2, ensure_ascii=False))
    print(f"\nError log saved to: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(n_samples: int = 100, seed: int = 42):
    tok, model, device = load_encoder()

    embs = np.load(INDEX_DIR / "embeddings.npy")
    meta = [json.loads(l) for l in (INDEX_DIR / "meta.jsonl").read_text().splitlines()]
    print(f"Index: {len(meta)} notes")

    print(f"\nBuilding test cases from {n_samples} sampled notes...")
    cases = build_test_cases(meta, n_samples=n_samples, seed=seed)
    print(f"Generated {len(cases)} cases across {len(INTENT_QUERIES)} intents.")

    results, errors = run_evaluation(cases, embs, meta, tok, model, device)

    print_metrics(results)
    print_errors(errors)
    save_errors(errors)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n_samples", type=int, default=100, help="Number of notes to sample")
    p.add_argument("--seed",      type=int, default=42)
    args = p.parse_args()
    main(n_samples=args.n_samples, seed=args.seed)
