"""Inspect source tables, key integrity, outcomes, missingness, and cutoff eligibility."""
import json
import pandas as pd
from common import DATA, SOURCE, OUT, METRICS, KEY, REQUIRED, CUTOFFS, ensure_dirs, require_columns, read_info_registration

def markdown_table(frame):
    cols=[str(c) for c in frame.columns]
    def cell(v): return "" if pd.isna(v) else str(v).replace("|","\\|").replace("\n"," ")
    lines=["| "+" | ".join(cols)+" |","| "+" | ".join(["---"]*len(cols))+" |"]
    lines.extend(["| "+" | ".join(cell(v) for v in row)+" |" for row in frame.itertuples(index=False,name=None)])
    return "\n".join(lines)

def run_inspection():
    ensure_dirs()
    rows, missing_rows, duplicate_rows, unique_rows = [], [], [], []
    for name, required in REQUIRED.items():
        path = SOURCE / name
        if not path.exists():
            raise FileNotFoundError(f"Required OULAD source file not found: {path}")
        use_dtype = {"id_student": "string"}
        df = pd.read_csv(path, dtype=use_dtype)
        require_columns(df, required, name)
        rows.append({"file": name, "rows": len(df), "columns": len(df.columns), "bytes": path.stat().st_size})
        for c in df.columns:
            missing_rows.append({"file": name, "column": c, "dtype": str(df[c].dtype), "missing_n": int(df[c].isna().sum()), "missing_rate": float(df[c].isna().mean())})
            unique_rows.append({"file": name, "column": c, "unique_n": int(df[c].nunique(dropna=True))})
        duplicate_rows.append({"file": name, "duplicate_full_rows": int(df.duplicated().sum()), "duplicate_enrollment_keys": int(df.duplicated(KEY).sum()) if name in {"studentInfo.csv","studentRegistration.csv"} else None})
    base = read_info_registration()
    outcomes = base.final_result.value_counts(dropna=False).rename_axis("final_result").reset_index(name="rows")
    outcomes.to_csv(METRICS / "original_outcomes.csv", index=False)
    excluded = []
    cohort_rows = []
    for k, end in CUTOFFS.items():
        reason = pd.Series("", index=base.index, dtype="string")
        def assign(mask, label):
            reason.loc[reason.eq("") & mask] = label
        assign(base.date_registration.isna(), "missing_registration_date")
        assign(base.date_registration.notna() & (base.date_registration > end), "registered_after_cutoff")
        assign(base.date_unregistration.notna() & (base.date_unregistration <= end), "unregistered_by_cutoff")
        assign((base.final_result == "Withdrawn") & base.date_unregistration.isna(), "withdrawn_timing_unknown")
        bad = base.loc[reason.ne(""), KEY + ["final_result"]].copy()
        bad["cutoff_days"] = k
        bad["exclusion_reason"] = reason.loc[reason.ne("")].to_numpy()
        excluded.append(bad)
        eligible = base.loc[reason.eq("")]
        cohort_rows.append({"cutoff_days": k, "cutoff_end_day_inclusive": end, "eligible_enrollments": len(eligible), "eligible_students": eligible.id_student.nunique(), "at_risk_prevalence": float(eligible.final_result.isin(["Fail", "Withdrawn"]).mean()), **{f"excluded_{x}": int((reason == x).sum()) for x in reason.unique() if x}})
    pd.DataFrame(rows).to_csv(METRICS / "input_file_inventory.csv", index=False)
    pd.DataFrame(missing_rows).to_csv(METRICS / "missing_values.csv", index=False)
    pd.DataFrame(duplicate_rows).to_csv(METRICS / "duplicate_checks.csv", index=False)
    pd.DataFrame(unique_rows).to_csv(METRICS / "unique_value_counts.csv", index=False)
    pd.DataFrame(cohort_rows).to_csv(METRICS / "cohort_eligibility.csv", index=False)
    pd.concat(excluded, ignore_index=True).to_csv(METRICS / "excluded_records.csv", index=False)
    md = ["# OULAD data quality report", "", "Source tables were read without modifying the input files.", "", "## Input inventory", "", markdown_table(pd.DataFrame(rows)), "", "## Outcome distribution", "", markdown_table(outcomes), "", "## Enrollment key and duplicate checks", "", markdown_table(pd.DataFrame(duplicate_rows)), "", "## Cutoff cohorts", "", markdown_table(pd.DataFrame(cohort_rows)), "", "Missing values and per-column types are in `missing_values.csv`; exclusions with their reasons are in `excluded_records.csv`."]
    (METRICS / "data_quality_report.md").write_text("\n".join(md), encoding="utf-8")
    print("Data inspection complete; source rows, schema, key quality, outcomes, and cutoff exclusions saved.")

if __name__ == "__main__": run_inspection()



