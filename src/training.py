"""Grouped model selection and student-disjoint train/validation/test evaluation."""
import json, hashlib, importlib.util
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.metrics import f1_score, precision_recall_fscore_support
from common import KEY, CUTOFFS, METRICS, PREDICTIONS, ROOT, OUTCOMES, SEED
from preprocess import assign_group_splits, save_split_report
from models import BASE, EXP, make_candidate
from evaluate import metric_dict

def candidates():
    names=["Dummy","LogisticRegression","DecisionTree","RandomForest","GradientBoosting","SoftVoting","Stacking"]
    if importlib.util.find_spec("xgboost"): names.append("XGBoost")
    if importlib.util.find_spec("lightgbm"): names.append("LightGBM")
    return names

def fit_cv_select(train, task, features):
    target="at_risk" if task=="binary" else "final_result"
    labels=[0,1] if task=="binary" else OUTCOMES
    X=train[features]; y=train[target].to_numpy(dtype=int if task=="binary" else str); groups=train.id_student.astype("string").to_numpy()
    n=min(3, len(np.unique(groups)))
    cv=GroupKFold(n_splits=n)
    records=[]
    for name in candidates():
        scores=[]
        for itr,iva in cv.split(X,y,groups):
            model=make_candidate(name,task,features,groups[itr] if name=="Stacking" else None)
            model.fit(X.iloc[itr],y[itr]); pred=model.predict(X.iloc[iva])
            scores.append(f1_score(y[iva],pred,average="binary",pos_label=1,zero_division=0) if task=="binary" else f1_score(y[iva],pred,average="macro",zero_division=0))
        records.append({"model":name,"grouped_cv_mean":float(np.mean(scores)),"grouped_cv_sd":float(np.std(scores,ddof=1)) if len(scores)>1 else 0.0,"fold_scores":json.dumps(scores)})
    table=pd.DataFrame(records).sort_values("grouped_cv_mean",ascending=False)
    table.to_csv(METRICS/f"{task}_grouped_cv_model_selection.csv",index=False)
    selected=table.iloc[0].model
    return selected,table

def fit_task(task, features_by_cutoff, assignments):
    target="at_risk" if task=="binary" else "final_result"
    labels=[0,1] if task=="binary" else OUTCOMES
    day28=features_by_cutoff.loc[features_by_cutoff.cutoff_days.eq(28)].copy()
    base=features_by_cutoff[KEY+[target]].drop_duplicates(KEY)
    save_split_report(base,assignments,target,task)
    assignments.to_csv(METRICS/f"{task}_split_assignments.csv",index=False)
    work=features_by_cutoff.merge(assignments[KEY+["split","group_fold"]],on=KEY,how="left",validate="many_to_one")
    if work.split.isna().any(): raise ValueError("Some enrollment did not receive a student-grouped split")
    train=work[(work.cutoff_days==28)&(work.split=="train")]
    selected,cvtable=fit_cv_select(train,task,EXP)
    full_metrics=[]; selected_predictions=[]; all_predictions=[]; models={}
    # Evaluate every candidate selected by training CV on day-28 expanded features.
    for name in candidates():
        estimator=make_candidate(name,task,EXP,train.id_student.to_numpy() if name=="Stacking" else None)
        estimator.fit(train[EXP],train[target])
        models[name]=estimator
        for split in ["validation","test"]:
            q=work[(work.cutoff_days==28)&(work.split==split)]
            y=q[target].to_numpy(dtype=int if task=="binary" else str)
            classes=list(estimator.classes_) if hasattr(estimator,"classes_") else list(estimator.named_steps["model"].classes_)
            raw=estimator.predict_proba(q[EXP]); prob=np.column_stack([raw[:,classes.index(c)] if c in classes else np.zeros(len(q)) for c in labels])
            pred=np.asarray(labels)[prob.argmax(axis=1)]
            if task=="binary": pred=(prob[:,1]>=.5).astype(int)
            m=metric_dict(y,pred,prob,labels); m.update({"task":task,"model":name,"selected_by_training_cv":name==selected,"split":split,"cutoff_days":28,"features":"expanded"})
            if split=="test": full_metrics.append(m)
            for i,(_,r) in enumerate(q.iterrows()):
                out={**{c:r[c] for c in KEY},"task":task,"model":name,"split":split,"cutoff_days":28,"features":"expanded","y_true":y[i],"y_pred":pred[i]}
                out.update({f"prob_{c}":float(prob[i,j]) for j,c in enumerate(labels)})
                all_predictions.append(out)
                if name==selected: selected_predictions.append(out)
    pd.DataFrame(full_metrics).to_csv(METRICS/f"{task}_model_comparison.csv",index=False)
    # Apply the training-CV-selected algorithm to VLE-only features as a same-split ablation.
    baseline_model=make_candidate(selected,task,BASE,train.id_student.to_numpy() if selected=="Stacking" else None)
    baseline_model.fit(train[BASE],train[target]); joblib.dump(baseline_model,ROOT/"models"/f"{task}_selected_vle_baseline.joblib")
    baseline_rows=[]
    for split in ["validation","test"]:
        q=work[(work.cutoff_days==28)&(work.split==split)]; y=q[target].to_numpy(dtype=int if task=="binary" else str)
        classes=list(baseline_model.classes_) if hasattr(baseline_model,"classes_") else list(baseline_model.named_steps["model"].classes_)
        raw=baseline_model.predict_proba(q[BASE]); prob=np.column_stack([raw[:,classes.index(c)] if c in classes else np.zeros(len(q)) for c in labels]); pred=np.asarray(labels)[prob.argmax(axis=1)]
        if task=="binary": pred=(prob[:,1]>=.5).astype(int)
        m=metric_dict(y,pred,prob,labels); m.update({"task":task,"model":selected,"selected_by_training_cv":True,"split":split,"cutoff_days":28,"features":"vle_baseline"}); baseline_rows.append(m)
        for i,(_,r) in enumerate(q.iterrows()):
            all_predictions.append({**{c:r[c] for c in KEY},"task":task,"model":selected,"split":split,"cutoff_days":28,"features":"vle_baseline","y_true":y[i],"y_pred":pred[i],**{f"prob_{c}":float(prob[i,j]) for j,c in enumerate(labels)}})
    base_test=pd.DataFrame([r for r in baseline_rows if r["split"]=="test"])
    pd.concat([pd.DataFrame(full_metrics),base_test],ignore_index=True).to_csv(METRICS/f"{task}_model_comparison.csv",index=False)
    # Evaluate the selected learner at all cutoffs, each with the same student assignment and feature ablation.
    cutoff_rows=[]
    for cutoff in CUTOFFS:
        qcut=work[work.cutoff_days==cutoff]
        for feature_name,cols in [("vle_baseline",BASE),("expanded",EXP)]:
            tr=qcut[qcut.split=="train"]; te=qcut[qcut.split=="test"]
            est=make_candidate(selected,task,cols,tr.id_student.to_numpy() if selected=="Stacking" else None)
            est.fit(tr[cols],tr[target]); y=te[target].to_numpy(dtype=int if task=="binary" else str)
            classes=list(est.classes_) if hasattr(est,"classes_") else list(est.named_steps["model"].classes_)
            raw=est.predict_proba(te[cols]); prob=np.column_stack([raw[:,classes.index(c)] if c in classes else np.zeros(len(te)) for c in labels]); pred=np.asarray(labels)[prob.argmax(axis=1)]
            if task=="binary": pred=(prob[:,1]>=.5).astype(int)
            mm=metric_dict(y,pred,prob,labels); mm.update({"task":task,"model":selected,"split":"test","cutoff_days":cutoff,"features":feature_name}); cutoff_rows.append(mm)
            if cutoff==28 and feature_name=="expanded": joblib.dump(est,ROOT/"models"/f"{task}_selected_expanded_day28.joblib")
    pd.DataFrame(cutoff_rows).to_csv(METRICS/f"{task}_cutoff_comparison.csv",index=False)
    sel_df=pd.DataFrame(selected_predictions)
    sel_df[sel_df.split=="test"].to_csv(PREDICTIONS/f"{task}_test_predictions.csv",index=False)
    sel_df[sel_df.split=="validation"].to_csv(PREDICTIONS/f"{task}_validation_predictions.csv",index=False)
    pd.DataFrame(all_predictions).to_csv(PREDICTIONS/f"{task}_all_model_predictions.csv",index=False)
    (METRICS/f"{task}_selected_model.json").write_text(json.dumps({"selected_model":selected,"selection_metric":"grouped cross-validation macro-F1" if task=="multiclass" else "grouped cross-validation F1","cv_splits":3,"feature_set_for_selection":"day-28 expanded","seed":SEED},indent=2),encoding="utf-8")
    return selected,work





