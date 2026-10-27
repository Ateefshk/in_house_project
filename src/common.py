"""Shared project configuration and validated OULAD schemas."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SOURCE = DATA / "original"
DERIVED = DATA / "derived"
OUT = ROOT / "outputs"
METRICS = OUT / "metrics"
PREDICTIONS = OUT / "predictions"
FIGURES = OUT / "figures"
SEED = 53
KEY = ["code_module", "code_presentation", "id_student"]
CUTOFFS = {7: 6, 14: 13, 21: 20, 28: 27}
OUTCOMES = ["Distinction", "Pass", "Fail", "Withdrawn"]

REQUIRED = {
    "studentInfo.csv": set(KEY + ["final_result", "num_of_prev_attempts", "studied_credits"]),
    "studentRegistration.csv": set(KEY + ["date_registration", "date_unregistration"]),
    "studentVle.csv": set(KEY + ["date", "sum_click"]),
    "assessments.csv": {"code_module", "code_presentation", "id_assessment", "date"},
    "studentAssessment.csv": {"id_assessment", "id_student", "date_submitted", "is_banked"},
    "courses.csv": {"code_module", "code_presentation", "module_presentation_length"},
    "vle.csv": {"id_site", "code_module", "activity_type", "week_from", "week_to"},
}

def ensure_dirs():
    for p in [METRICS, PREDICTIONS, FIGURES, ROOT / "models"]:
        p.mkdir(parents=True, exist_ok=True)

def require_columns(frame, columns, label):
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")

def read_info_registration():
    info = pd.read_csv(SOURCE / "studentInfo.csv", dtype={"id_student": "string"})
    reg = pd.read_csv(SOURCE / "studentRegistration.csv", dtype={"id_student": "string"})
    require_columns(info, REQUIRED["studentInfo.csv"], "studentInfo.csv")
    require_columns(reg, REQUIRED["studentRegistration.csv"], "studentRegistration.csv")
    if info.duplicated(KEY).any() or reg.duplicated(KEY).any():
        raise ValueError("Duplicate enrollment keys found in studentInfo/studentRegistration")
    base = info.merge(reg, on=KEY, how="left", validate="one_to_one", indicator=True)
    if not base._merge.eq("both").all():
        raise ValueError("Some studentInfo enrollment keys do not match studentRegistration")
    base = base.drop(columns="_merge")
    for c in ["date_registration", "date_unregistration"]:
        base[c] = pd.to_numeric(base[c], errors="coerce")
    return base

