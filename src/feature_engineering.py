"""Build leakage-safe cutoff features from OULAD event-day data and assessment events."""
import numpy as np
import pandas as pd
from common import SOURCE, DERIVED, OUT, METRICS, KEY, CUTOFFS, require_columns, read_info_registration

def read_daily_activity():
    path = SOURCE / "studentVle.csv"
    require_columns(pd.read_csv(path, nrows=0), KEY + ["date", "sum_click"], "studentVle.csv")
    daily=[]
    for chunk in pd.read_csv(path, usecols=KEY + ["date", "sum_click"], dtype={"id_student":"string"}, chunksize=750_000):
        chunk["date"] = pd.to_numeric(chunk.date, errors="coerce")
        chunk["sum_click"] = pd.to_numeric(chunk.sum_click, errors="coerce").fillna(0)
        chunk = chunk.loc[chunk.date.between(0, 27)]
        if not chunk.empty:
            daily.append(chunk.groupby(KEY + ["date"], as_index=False, observed=True).sum_click.sum())
    if not daily:
        raise ValueError("studentVle.csv contains no dated VLE events in days 0-27")
    d=pd.concat(daily, ignore_index=True).groupby(KEY + ["date"], as_index=False, observed=True).sum_click.sum()
    d["active_day"]=(d.sum_click > 0).astype("int8")
    return d

def cutoff_exclusions(base, end):
    reason=pd.Series("", index=base.index, dtype="string")
    def assign(mask, label):
        reason.loc[reason.eq("") & mask] = label
    assign(base.date_registration.isna(), "missing_registration_date")
    assign(base.date_registration.notna() & (base.date_registration > end), "registered_after_cutoff")
    assign(base.date_unregistration.notna() & (base.date_unregistration <= end), "unregistered_by_cutoff")
    assign((base.final_result == "Withdrawn") & base.date_unregistration.isna(), "withdrawn_timing_unknown")
    return reason

def assessment_features(cohort, cutoff):
    end=CUTOFFS[cutoff]
    ass=pd.read_csv(SOURCE / "assessments.csv")
    sub=pd.read_csv(SOURCE / "studentAssessment.csv", dtype={"id_student":"string"})
    require_columns(ass, ["code_module","code_presentation","id_assessment","date"], "assessments.csv")
    require_columns(sub, ["id_assessment","id_student","date_submitted","is_banked"], "studentAssessment.csv")
    ass["date"]=pd.to_numeric(ass.date, errors="coerce")
    sub["date_submitted"]=pd.to_numeric(sub.date_submitted, errors="coerce")
    sub=sub.loc[sub.is_banked.eq(0)]
    due=ass.loc[ass.date.between(0, end), ["code_module","code_presentation","id_assessment","date"]]
    due_student=cohort[KEY+["date_registration"]].merge(due, on=["code_module","code_presentation"], how="left")
    due_student=due_student.loc[due_student.date.notna() & (due_student.date >= due_student.date_registration)]
    due_count=due_student.groupby(KEY).id_assessment.nunique().rename("assessments_due")
    due_full=cohort[KEY].merge(due_count, on=KEY, how="left")
    due_full["assessments_due"]=due_full.assessments_due.fillna(0).astype(int)
    events=ass.merge(sub, on="id_assessment", how="inner", validate="one_to_many")
    events=events.loc[events.date.between(0, end) & events.date_submitted.between(0, end)]
    observed=cohort[KEY+["date_registration"]].merge(events[["code_module","code_presentation","id_student","id_assessment","date","date_submitted"]], on=KEY, how="left")
    observed=observed.loc[observed.id_assessment.notna() & (observed.date >= observed.date_registration) & (observed.date_submitted >= observed.date_registration)]
    sub_count=observed.groupby(KEY).id_assessment.nunique().rename("assessments_submitted")
    result=due_full.merge(sub_count, on=KEY, how="left")
    result["assessments_submitted"]=result.assessments_submitted.fillna(0).astype(int)
    result["assessments_missed"]=(result.assessments_due-result.assessments_submitted).clip(lower=0)
    return result

def build_features():
    base=read_info_registration()
    daily=read_daily_activity()
    windows=[]; excluded=[]; cohort_audit=[]
    for cutoff,end in CUTOFFS.items():
        reason=cutoff_exclusions(base,end)
        bad=base.loc[reason.ne(""), KEY+["final_result"]].copy()
        bad["cutoff_days"]=cutoff; bad["exclusion_reason"]=reason.loc[reason.ne("")].to_numpy(); excluded.append(bad)
        cohort=base.loc[reason.eq("")].copy()
        cohort_audit.append({"cutoff_days":cutoff,"cutoff_end_day_inclusive":end,"source_enrollments":len(base),"eligible_enrollments":len(cohort),"eligible_students":cohort.id_student.nunique(),"at_risk_prevalence":float(cohort.final_result.isin(["Fail","Withdrawn"]).mean()),**{f"excluded_{x}":int((reason==x).sum()) for x in reason.unique() if x}})
        window=daily.loc[daily.date <= end]
        total=window.groupby(KEY, observed=True).agg(clicks=("sum_click","sum"),active_days=("active_day","sum")).reset_index()
        recent_start=max(0,end-6)
        recent=window.loc[window.date >= recent_start].groupby(KEY, observed=True).agg(clicks_last7=("sum_click","sum"),active_days_last7=("active_day","sum")).reset_index()
        feat=cohort.merge(total,on=KEY,how="left",validate="one_to_one").merge(recent,on=KEY,how="left",validate="one_to_one")
        for c in ["clicks","active_days","clicks_last7","active_days_last7"]: feat[c]=feat[c].fillna(0)
        feat["cutoff_days"]=cutoff
        feat["enrolled_days_by_cutoff"]=(cutoff-np.maximum(feat.date_registration.fillna(0),0)).clip(lower=1)
        feat["clicks_per_enrolled_day"]=feat.clicks/feat.enrolled_days_by_cutoff
        feat["active_day_rate"]=feat.active_days/feat.enrolled_days_by_cutoff
        feat["recent_click_share"]=feat.clicks_last7/feat.clicks.replace(0,np.nan)
        feat["recent_click_share"]=feat.recent_click_share.fillna(0)
        feat["log_clicks"]=np.log1p(feat.clicks)
        feat["at_risk"]=feat.final_result.isin(["Fail","Withdrawn"]).astype("int8")
        feat=feat.merge(assessment_features(feat,cutoff),on=KEY,how="left",validate="one_to_one")
        windows.append(feat)
    output=pd.concat(windows,ignore_index=True)
    METRICS.mkdir(parents=True,exist_ok=True)
    output.to_csv(OUT/"features_by_cutoff.csv",index=False)
    pd.concat(excluded,ignore_index=True).to_csv(METRICS/"excluded_records.csv",index=False)
    pd.DataFrame(cohort_audit).to_csv(METRICS/"cohort_eligibility.csv",index=False)
    verify_derivative(daily)
    return output

def verify_derivative(daily):
    path=DERIVED/"oulad_first28days.csv"
    if not path.exists():
        print("Published derivative unavailable; reconstruction comparison skipped.")
        return
    published=pd.read_csv(path,dtype={"id_student":"string"})
    rebuilt=daily.groupby(KEY,observed=True).agg(rebuilt_clicks=("sum_click","sum"),rebuilt_active_days=("active_day","sum")).reset_index()
    check=published.merge(rebuilt,on=KEY,how="left",validate="one_to_one")
    check[["rebuilt_clicks","rebuilt_active_days"]]=check[["rebuilt_clicks","rebuilt_active_days"]].fillna(0)
    result={"published_rows":len(published),"key_rows_match":len(check)==len(published),"clicks_match":bool((check.vle_clicks_days_0_27==check.rebuilt_clicks).all()),"active_days_match":bool((check.vle_active_days_0_27==check.rebuilt_active_days).all()),"zero_activity_enrollments":int(check.rebuilt_clicks.eq(0).sum())}
    pd.DataFrame([result]).to_csv(METRICS/"source_reconstruction_check.csv",index=False)
    if not all([result["key_rows_match"],result["clicks_match"],result["active_days_match"]]):
        raise AssertionError(f"Reconstructed 28-day activity does not match published derivative: {result}")
    print("Source reconstruction matches the published 0-27 day derivative.")

if __name__ == "__main__":
    from common import ensure_dirs
    ensure_dirs(); build_features()

