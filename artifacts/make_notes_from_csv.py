import json
from pathlib import Path
from typing import Optional

import pandas as pd

CSV_DIR = Path("data/synthea_csv")
OUT_DIR = Path("data/corpus")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_JSONL = OUT_DIR / "synthea_notes.jsonl"
OUT_TXT = OUT_DIR / "all_notes_text.txt"

MAX_MEDICATIONS = 50
MAX_PROCEDURES = 50
MAX_OBSERVATIONS = 100
MAX_CONDITIONS = 50


def load_csv(filename: str) -> pd.DataFrame:
    path = CSV_DIR / filename
    if not path.exists():
        print(f"[WARN] Missing file: {filename} (skipping)")
        return pd.DataFrame()
    return pd.read_csv(path)


def standardize_ids(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    if "Id" in df.columns:
        if "PATIENT" in df.columns:
            df = df.rename(columns={"Id": "ENCOUNTER_ID"})
        else:
            df = df.rename(columns={"Id": "PATIENT_ID"})

    if "PATIENT" in df.columns:
        df = df.rename(columns={"PATIENT": "PATIENT_ID"})

    if "ENCOUNTER" in df.columns:
        df = df.rename(columns={"ENCOUNTER": "ENCOUNTER_ID"})

    return df


def safe(val, default: str = "") -> str:
    # float('nan') is truthy in Python, so `if val:` won't catch it — always wrap DataFrame values with this
    if val is None:
        return default
    try:
        import math
        if isinstance(val, float) and math.isnan(val):
            return default
    except (TypeError, ValueError):
        pass
    s = str(val).strip()
    return s if s not in ("nan", "NaN", "None", "") else default


def append_detail(parts: list[str], label: str, value: str) -> None:
    value = safe(value)
    if value:
        parts.append(f"{label}: {value}")


def format_item(title: str, *details: tuple[str, str]) -> str:
    parts = [safe(title, "Unknown")]
    for label, value in details:
        append_detail(parts, label, value)
    return " | ".join(parts)


def build_note_for_encounter(
    enc_row: pd.Series,
    patient_row: Optional[pd.Series],
    cond_df: pd.DataFrame,
    med_df: pd.DataFrame,
    obs_df: pd.DataFrame,
    proc_df: pd.DataFrame,
) -> str:
    lines = []

    lines.append("DISCHARGE SUMMARY")

    encounter_id = safe(enc_row.get("ENCOUNTER_ID", ""))
    patient_id   = safe(enc_row.get("PATIENT_ID", ""))
    start        = safe(enc_row.get("START", ""))
    stop         = safe(enc_row.get("STOP", ""))
    eclass       = safe(enc_row.get("ENCOUNTERCLASS", enc_row.get("CLASS", "")))
    description  = safe(enc_row.get("DESCRIPTION", ""))
    encounter_code = safe(enc_row.get("CODE", ""))
    reason_code  = safe(enc_row.get("REASONCODE", ""))
    reason_desc  = safe(enc_row.get("REASONDESCRIPTION", ""))
    organization = safe(enc_row.get("ORGANIZATION", ""))
    provider     = safe(enc_row.get("PROVIDER", ""))
    payer        = safe(enc_row.get("PAYER", ""))
    base_cost    = safe(enc_row.get("BASE_ENCOUNTER_COST", ""))
    claim_cost   = safe(enc_row.get("TOTAL_CLAIM_COST", ""))
    coverage     = safe(enc_row.get("PAYER_COVERAGE", ""))

    lines.append(f"Encounter ID: {encounter_id}")
    lines.append(f"Patient ID: {patient_id}")
    if start or stop:
        lines.append(f"Dates: {start} to {stop}")
    if eclass:
        lines.append(f"Encounter Class: {eclass}")
    if description:
        lines.append(f"Visit Reason: {description}")
    if encounter_code:
        lines.append(f"Encounter Code: {encounter_code}")
    if reason_desc:
        lines.append(f"Reason Description: {reason_desc}" + (f" | {reason_code}" if reason_code else ""))
    if organization:
        lines.append(f"Organization ID: {organization}")
    if provider:
        lines.append(f"Provider ID: {provider}")
    if payer:
        lines.append(f"Payer ID: {payer}")
    if base_cost or claim_cost or coverage:
        cost_parts = []
        if base_cost:
            cost_parts.append(f"base encounter cost: {base_cost}")
        if claim_cost:
            cost_parts.append(f"total claim cost: {claim_cost}")
        if coverage:
            cost_parts.append(f"payer coverage: {coverage}")
        lines.append("Financials: " + " | ".join(cost_parts))
    lines.append("")

    if patient_row is not None:
        prefix    = safe(patient_row.get("PREFIX", ""))
        first     = safe(patient_row.get("FIRST", ""))
        middle    = safe(patient_row.get("MIDDLE", ""))
        last      = safe(patient_row.get("LAST", ""))
        suffix    = safe(patient_row.get("SUFFIX", ""))
        full_name = " ".join(p for p in [prefix, first, middle, last, suffix] if p)

        gender    = safe(patient_row.get("GENDER", ""), "unknown")
        race      = safe(patient_row.get("RACE", ""), "unknown")
        ethnicity = safe(patient_row.get("ETHNICITY", ""), "unknown")
        birthdate = safe(patient_row.get("BIRTHDATE", ""))
        deathdate = safe(patient_row.get("DEATHDATE", ""))
        marital   = safe(patient_row.get("MARITAL", ""))
        birthplace = safe(patient_row.get("BIRTHPLACE", ""))
        address   = safe(patient_row.get("ADDRESS", ""))
        city      = safe(patient_row.get("CITY", ""))
        state     = safe(patient_row.get("STATE", ""))
        zip_code  = safe(patient_row.get("ZIP", ""))
        county    = safe(patient_row.get("COUNTY", ""))
        expenses  = safe(patient_row.get("HEALTHCARE_EXPENSES", ""))
        coverage_amt = safe(patient_row.get("HEALTHCARE_COVERAGE", ""))
        income    = safe(patient_row.get("INCOME", ""))

        lines.append("DEMOGRAPHICS:")
        if full_name:
            lines.append(f"- Name: {full_name}")
        lines.append(f"- Sex: {gender}")
        lines.append(f"- Race: {race}")
        lines.append(f"- Ethnicity: {ethnicity}")
        lines.append(f"- Birthdate: {birthdate}")
        if deathdate:
            lines.append(f"- Deathdate: {deathdate}")
        if marital:
            lines.append(f"- Marital Status: {marital}")
        if birthplace:
            lines.append(f"- Birthplace: {birthplace}")
        location = ", ".join(part for part in [address, city, state, zip_code] if part)
        if location:
            lines.append(f"- Address: {location}")
        if county:
            lines.append(f"- County: {county}")
        if expenses or coverage_amt or income:
            finance_bits = []
            if expenses:
                finance_bits.append(f"healthcare expenses: {expenses}")
            if coverage_amt:
                finance_bits.append(f"healthcare coverage: {coverage_amt}")
            if income:
                finance_bits.append(f"income: {income}")
            lines.append("- Financial Profile: " + " | ".join(finance_bits))
        lines.append("")

    # diagnoses go first so they survive the 256-token embedding window
    if not cond_df.empty:
        lines.append("DIAGNOSES / CONDITIONS:")
        for _, r in cond_df.head(MAX_CONDITIONS).iterrows():
            lines.append("- " + format_item(
                safe(r.get("DESCRIPTION", "")),
                ("code", r.get("CODE", "")),
                ("onset", r.get("START", "")),
                ("resolved", r.get("STOP", "")),
            ))
        lines.append("")

    if not med_df.empty:
        lines.append("MEDICATIONS:")
        for _, r in med_df.head(MAX_MEDICATIONS).iterrows():
            lines.append("- " + format_item(
                safe(r.get("DESCRIPTION", "")),
                ("code", r.get("CODE", "")),
                ("start", r.get("START", "")),
                ("stop", r.get("STOP", "")),
                ("dispenses", r.get("DISPENSES", "")),
                ("reason", r.get("REASONDESCRIPTION", "")),
            ))
        lines.append("")

    if not proc_df.empty:
        lines.append("PROCEDURES:")
        for _, r in proc_df.head(MAX_PROCEDURES).iterrows():
            lines.append("- " + format_item(
                safe(r.get("DESCRIPTION", "")),
                ("code", r.get("CODE", "")),
                ("start", r.get("START", "")),
                ("reason", r.get("REASONDESCRIPTION", "")),
            ))
        lines.append("")

    if not obs_df.empty:
        lines.append("LABS / OBSERVATIONS:")
        date_col = "DATE" if "DATE" in obs_df.columns else ("START" if "START" in obs_df.columns else None)
        for _, r in obs_df.head(MAX_OBSERVATIONS).iterrows():
            value = safe(r.get("VALUE", ""))
            units = safe(r.get("UNITS", ""))
            measured = f"{value} {units}".strip()
            lines.append("- " + format_item(
                safe(r.get("DESCRIPTION", "")),
                ("value", measured),
                ("date", safe(r.get(date_col, "")) if date_col else ""),
                ("category", r.get("CATEGORY", "")),
                ("code", r.get("CODE", "")),
            ))
        lines.append("")

    lines.append("PLAN:")
    lines.append("- Follow-up with primary care.")
    lines.append("- Continue medications as prescribed.")
    lines.append("- Return for worsening symptoms.")

    return "\n".join(lines).strip()


def main():
    patients      = standardize_ids(load_csv("patients.csv"))
    encounters    = standardize_ids(load_csv("encounters.csv"))
    conditions    = standardize_ids(load_csv("conditions.csv"))
    medications   = standardize_ids(load_csv("medications.csv"))
    observations  = standardize_ids(load_csv("observations.csv"))
    procedures    = standardize_ids(load_csv("procedures.csv"))

    if patients.empty or encounters.empty:
        raise RuntimeError("You must have patients.csv and encounters.csv in data/synthea_csv/")

    patients = patients.set_index("PATIENT_ID", drop=False)

    cond_by_enc = conditions.groupby("ENCOUNTER_ID") if not conditions.empty else None
    med_by_enc  = medications.groupby("ENCOUNTER_ID") if not medications.empty else None
    obs_by_enc  = observations.groupby("ENCOUNTER_ID") if not observations.empty else None
    proc_by_enc = procedures.groupby("ENCOUNTER_ID") if not procedures.empty else None
    # fallback: use patient-level conditions for encounters that have none linked
    cond_by_pat = conditions.groupby("PATIENT_ID") if not conditions.empty else None

    written = 0

    with OUT_JSONL.open("w", encoding="utf-8") as f_json, OUT_TXT.open("w", encoding="utf-8") as f_txt:
        for _, enc_row in encounters.iterrows():
            enc_id = enc_row.get("ENCOUNTER_ID")
            pat_id = enc_row.get("PATIENT_ID")

            patient_row = patients.loc[pat_id] if pat_id in patients.index else None

            cond_df = (
                cond_by_enc.get_group(enc_id)
                if cond_by_enc is not None and enc_id in cond_by_enc.groups
                else pd.DataFrame()
            )
            if cond_df.empty and cond_by_pat is not None and pat_id in cond_by_pat.groups:
                cond_df = cond_by_pat.get_group(pat_id)

            med_df  = med_by_enc.get_group(enc_id)  if med_by_enc  is not None and enc_id in med_by_enc.groups  else pd.DataFrame()
            obs_df  = obs_by_enc.get_group(enc_id)  if obs_by_enc  is not None and enc_id in obs_by_enc.groups  else pd.DataFrame()
            proc_df = proc_by_enc.get_group(enc_id) if proc_by_enc is not None and enc_id in proc_by_enc.groups else pd.DataFrame()

            note_text = build_note_for_encounter(
                enc_row,
                patient_row,
                cond_df,
                med_df,
                obs_df,
                proc_df,
            )

            if len(note_text) < 200:
                continue

            f_json.write(json.dumps({
                "encounter_id": enc_id,
                "patient_id":   pat_id,
                "text":         note_text,
            }, ensure_ascii=False) + "\n")
            f_txt.write(note_text + "\n\n")
            written += 1

    print(f"Done. Wrote {written} notes to: {OUT_JSONL}")
    print(f"Plain text corpus saved to: {OUT_TXT}")


if __name__ == "__main__":
    main()
