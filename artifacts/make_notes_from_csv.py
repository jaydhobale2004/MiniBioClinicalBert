import json
from pathlib import Path

import pandas as pd

# ----------------------------
# Paths
# ----------------------------
CSV_DIR = Path("data/synthea_csv")
OUT_DIR = Path("data/corpus")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_JSONL = OUT_DIR / "synthea_notes.jsonl"


# ----------------------------
# Helper: load CSV safely
# ----------------------------
def load_csv(filename: str) -> pd.DataFrame:
    path = CSV_DIR / filename
    if not path.exists():
        print(f"[WARN] Missing file: {filename} (skipping)")
        return pd.DataFrame()
    return pd.read_csv(path)


# ----------------------------
# Helper: standardize column names
# Synthea commonly uses:
#   - patients.csv: Id
#   - encounters.csv: Id, PATIENT
#   - other tables: PATIENT, ENCOUNTER
# We rename them so our code is consistent.
# ----------------------------
def standardize_ids(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    # Rename "Id" depending on which file it likely is.
    if "Id" in df.columns:
        # If it has PATIENT column, it's probably encounters.csv
        # otherwise likely patients.csv
        if "PATIENT" in df.columns:
            df = df.rename(columns={"Id": "ENCOUNTER_ID"})
        else:
            df = df.rename(columns={"Id": "PATIENT_ID"})

    if "PATIENT" in df.columns:
        df = df.rename(columns={"PATIENT": "PATIENT_ID"})

    if "ENCOUNTER" in df.columns:
        df = df.rename(columns={"ENCOUNTER": "ENCOUNTER_ID"})

    return df


# ----------------------------
# Build ONE note text for ONE encounter
# ----------------------------
def build_note_for_encounter(enc_row: pd.Series,
                             patient_row: pd.Series | None,
                             cond_df: pd.DataFrame,
                             med_df: pd.DataFrame,
                             obs_df: pd.DataFrame,
                             proc_df: pd.DataFrame) -> str:
    lines = []

    # Title
    lines.append("DISCHARGE SUMMARY")

    # Encounter basics
    encounter_id = enc_row.get("ENCOUNTER_ID", "")
    patient_id = enc_row.get("PATIENT_ID", "")
    start = enc_row.get("START", "")
    stop = enc_row.get("STOP", "")
    eclass = enc_row.get("ENCOUNTERCLASS", enc_row.get("CLASS", ""))
    description = enc_row.get("DESCRIPTION", "")

    lines.append(f"Encounter ID: {encounter_id}")
    lines.append(f"Patient ID: {patient_id}")
    if start or stop:
        lines.append(f"Dates: {start} to {stop}")
    if eclass:
        lines.append(f"Encounter Class: {eclass}")
    if description:
        lines.append(f"Reason/Description: {description}")
    lines.append("")

    # Demographics from patients.csv
    if patient_row is not None:
        gender = patient_row.get("GENDER", "unknown")
        race = patient_row.get("RACE", "unknown")
        ethnicity = patient_row.get("ETHNICITY", "unknown")
        birthdate = patient_row.get("BIRTHDATE", "")

        lines.append("DEMOGRAPHICS:")
        lines.append(f"- Sex: {gender}")
        lines.append(f"- Race: {race}")
        lines.append(f"- Ethnicity: {ethnicity}")
        lines.append(f"- Birthdate: {birthdate}")
        lines.append("")

    # Conditions / diagnoses
    if not cond_df.empty:
        lines.append("DIAGNOSES / CONDITIONS:")
        # We only take first N rows to keep notes reasonable
        for _, r in cond_df.head(30).iterrows():
            name = r.get("DESCRIPTION", "")
            code = r.get("CODE", "")
            cstart = r.get("START", "")
            text = f"- {name}"
            if code:
                text += f" | {code}"
            if cstart:
                text += f" | onset: {cstart}"
            lines.append(text)
        lines.append("")

    # Medications
    if not med_df.empty:
        lines.append("MEDICATIONS:")
        for _, r in med_df.head(30).iterrows():
            name = r.get("DESCRIPTION", "")
            code = r.get("CODE", "")
            mstart = r.get("START", "")
            text = f"- {name}"
            if code:
                text += f" | {code}"
            if mstart:
                text += f" | start: {mstart}"
            lines.append(text)
        lines.append("")

    # Procedures
    if not proc_df.empty:
        lines.append("PROCEDURES:")
        for _, r in proc_df.head(30).iterrows():
            name = r.get("DESCRIPTION", "")
            code = r.get("CODE", "")
            pstart = r.get("START", "")
            text = f"- {name}"
            if code:
                text += f" | {code}"
            if pstart:
                text += f" | date: {pstart}"
            lines.append(text)
        lines.append("")

    # Observations (can be huge, so sample a limited number)
    if not obs_df.empty:
        lines.append("LABS / OBSERVATIONS:")

        # Different Synthea versions may store dates in DATE or START
        date_col = "DATE" if "DATE" in obs_df.columns else ("START" if "START" in obs_df.columns else None)

        for _, r in obs_df.head(60).iterrows():
            name = r.get("DESCRIPTION", "")
            value = r.get("VALUE", "")
            units = r.get("UNITS", "")
            d = r.get(date_col, "") if date_col else ""

            text = f"- {name}: {value} {units}".strip()
            if d:
                text += f" | {d}"
            lines.append(text)

        lines.append("")

    # Add a small generic plan (makes notes more “note-like”)
    lines.append("PLAN:")
    lines.append("- Follow-up with primary care.")
    lines.append("- Continue medications as prescribed.")
    lines.append("- Return for worsening symptoms.")

    return "\n".join(lines).strip()


def main():
    # Load CSV files
    patients = standardize_ids(load_csv("patients.csv"))
    encounters = standardize_ids(load_csv("encounters.csv"))
    conditions = standardize_ids(load_csv("conditions.csv"))
    medications = standardize_ids(load_csv("medications.csv"))
    observations = standardize_ids(load_csv("observations.csv"))
    procedures = standardize_ids(load_csv("procedures.csv"))

    if patients.empty or encounters.empty:
        raise RuntimeError("You must have patients.csv and encounters.csv in data/synthea_csv/")

    # Make patient lookup (fast access by patient_id)
    patients = patients.set_index("PATIENT_ID", drop=False)

    # Group the tables by encounter for quick slicing
    cond_by_enc = conditions.groupby("ENCOUNTER_ID") if not conditions.empty else None
    med_by_enc = medications.groupby("ENCOUNTER_ID") if not medications.empty else None
    obs_by_enc = observations.groupby("ENCOUNTER_ID") if not observations.empty else None
    proc_by_enc = procedures.groupby("ENCOUNTER_ID") if not procedures.empty else None
    cond_by_pat = conditions.groupby("PATIENT_ID") if not conditions.empty else None


    written = 0

    with OUT_JSONL.open("w", encoding="utf-8") as f:
        for _, enc_row in encounters.iterrows():
            enc_id = enc_row.get("ENCOUNTER_ID")
            pat_id = enc_row.get("PATIENT_ID")

            patient_row = patients.loc[pat_id] if pat_id in patients.index else None

            # Get rows related to this encounter (or empty DataFrame)
            cond_df = (
                cond_by_enc.get_group(enc_id)
                if cond_by_enc is not None and enc_id in cond_by_enc.groups
                else pd.DataFrame()
            )

# Option A fallback: if this encounter has no conditions, use patient-level conditions
            if cond_df.empty and cond_by_pat is not None and pat_id in cond_by_pat.groups:
                cond_df = cond_by_pat.get_group(pat_id)
            med_df = med_by_enc.get_group(enc_id) if med_by_enc is not None and enc_id in med_by_enc.groups else pd.DataFrame()
            obs_df = obs_by_enc.get_group(enc_id) if obs_by_enc is not None and enc_id in obs_by_enc.groups else pd.DataFrame()
            proc_df = proc_by_enc.get_group(enc_id) if proc_by_enc is not None and enc_id in proc_by_enc.groups else pd.DataFrame()

            note_text = build_note_for_encounter(enc_row, patient_row, cond_df, med_df, obs_df, proc_df)

            # Simple filter: skip tiny notes
            if len(note_text) < 200:
                continue

            obj = {
                "encounter_id": enc_id,
                "patient_id": pat_id,
                "text": note_text
            }

            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            written += 1

    print(f"Done. Wrote {written} notes to: {OUT_JSONL}")


if __name__ == "__main__":
    main()
