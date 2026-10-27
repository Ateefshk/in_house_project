"""Held-out permutation importance, masked prediction examples, and cluster intervals."""
import os
os.environ.setdefault("MPLBACKEND", "Agg")
import hashlib
import importlib.util
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score, balanced_accuracy_score, precision_recall_fscore_support
from common import ROOT, METRICS, PREDICTIONS, FIGURES, OUTCOMES, SEED, KEY
from models import BASE, EXP

def cluster_bootstrap(frame,task,reps=300):
    labels=[0,1] if task=="binary" else OUTCOMES; y=frame.y_true.to_numpy(dtype=int if task=="binary" else str); pred=frame.y_pred.to_numpy(dtype=int if task=="binary" else str)
    ids,group_code=np.unique(frame.id_student.astype("string").to_numpy(),return_inverse=True); rng=np.random.default_rng(SEED)
    metric_values={"macro_f1":[],"weighted_f1":[],"balanced_accuracy":[]}
    if task=="binary": metric_values.update({"at_risk_recall":[],"at_risk_precision":[]})
    for _ in range(reps):
        counts=np.bincount(rng.integers(0,len(ids),size=len(ids)),minlength=len(ids)); weight=counts[group_code]
        try:
            if any(weight[y==c].sum()==0 for c in labels): continue
            metric_values["macro_f1"].append(f1_score(y,pred,labels=labels,average="macro",sample_weight=weight,zero_division=0))
            metric_values["weighted_f1"].append(f1_score(y,pred,labels=labels,average="weighted",sample_weight=weight,zero_division=0))
            metric_values["balanced_accuracy"].append(balanced_accuracy_score(y,pred,sample_weight=weight))
            if task=="binary":
                pr,rc,_,_=precision_recall_fscore_support(y,pred,average="binary",sample_weight=weight,zero_division=0)
                metric_values["at_risk_recall"].append(rc); metric_values["at_risk_precision"].append(pr)
        except ValueError:
            continue
    rows=[]
    for metric,vals in metric_values.items():
        rows.append({"task":task,"metric":metric,"point_estimate":float(np.mean(vals)),"ci_2_5":float(np.quantile(vals,.025)),"ci_97_5":float(np.quantile(vals,.975)),"requested_replicates":reps,"valid_replicates":len(vals),"seed":SEED,"unit":"student cluster"})
    return rows

def run_explainability():
    importance=[]; boot=[]
    for task in ["binary","multiclass"]:
        model_path=ROOT/"models"/f"{task}_selected_expanded_day28.joblib"
        pred_path=PREDICTIONS/f"{task}_test_predictions.csv"
        if not model_path.exists() or not pred_path.exists(): raise FileNotFoundError(f"Missing selected model/predictions for {task}")
        model=joblib.load(model_path); pred_df=pd.read_csv(pred_path,dtype={"id_student":"string"})
        features=EXP; source=pd.read_csv(ROOT/"outputs"/"features_by_cutoff.csv",dtype={"id_student":"string"})
        test=source[(source.cutoff_days==28)&source.set_index(["code_module","code_presentation","id_student"]).index.isin(pred_df.set_index(["code_module","code_presentation","id_student"]).index)].copy()
        # Join exact prediction keys and keep the feature input order.
        test=pred_df[["code_module","code_presentation","id_student"]].merge(test,on=["code_module","code_presentation","id_student"],validate="one_to_one")
        target="at_risk" if task=="binary" else "final_result"; y=test[target].to_numpy(dtype=int if task=="binary" else str)
        scoring="f1" if task=="binary" else "f1_macro"
        result=permutation_importance(model,test[features],y,n_repeats=8,random_state=SEED,scoring=scoring,n_jobs=1)
        for f,m,sd in zip(features,result.importances_mean,result.importances_std): importance.append({"task":task,"feature":f,"importance_mean":float(m),"importance_sd":float(sd),"test_rows":len(test),"scoring":scoring})
        # Pseudonymize IDs and omit enrollment keys from the deliverable examples.
        pcols=[f"prob_{x}" for x in ([0,1] if task=="binary" else OUTCOMES)]
        examples=pred_df.sample(min(12,len(pred_df)),random_state=SEED).copy()
        examples["example_id"]=[hashlib.sha256((str(r.id_student)+str(i)+"masked-case").encode()).hexdigest()[:12] for i,r in examples.iterrows()]
        target_col="at_risk" if task=="binary" else "final_result"
        detail=examples[["example_id"]+KEY+ ["y_true","y_pred"]+pcols].merge(test[KEY+[f for f in features if f not in KEY]+[target_col]],on=KEY,how="left",validate="one_to_one")
        detail=detail.rename(columns={target_col:"observed_outcome"}).drop(columns=KEY)
        merged=detail
        merged.to_csv(PREDICTIONS/f"{task}_masked_example_predictions.csv",index=False)
        boot.extend(cluster_bootstrap(pred_df,task,300))
        # Explain optional SHAP availability without making the project depend on it.
        if importlib.util.find_spec("shap"):
            try:
                import shap
                background=test[features].sample(min(40,len(test)),random_state=SEED)
                sample=test[features].sample(min(30,len(test)),random_state=SEED+1)
                explainer=shap.Explainer(model.predict_proba,background)
                values=explainer(sample)
                np.savez_compressed(METRICS/f"{task}_shap_sample.npz",values=np.asarray(values.values),base_values=np.asarray(values.base_values))
                shap_state=f"SHAP calculation completed for {task} on a small sample."
            except Exception as exc: shap_state=f"SHAP installed but incompatible or failed for {task}: {type(exc).__name__}: {exc}"
        else: shap_state="SHAP unavailable; permutation importance is used."
        (METRICS/f"{task}_shap_status.txt").write_text(shap_state,encoding="utf-8")
        top=sorted([z for z in importance if z["task"]==task],key=lambda z:z["importance_mean"],reverse=True)
        fig,ax=plt.subplots(figsize=(7,5)); show=top[:12][::-1]; ax.barh([z["feature"] for z in show],[z["importance_mean"] for z in show],xerr=[z["importance_sd"] for z in show],color="#376996"); ax.set(xlabel=f"Permutation decrease in {scoring}",title=f"{task.title()} held-out feature importance"); fig.tight_layout(); fig.savefig(FIGURES/f"{task}_feature_importance.png",dpi=220); plt.close(fig)
    pd.DataFrame(importance).to_csv(METRICS/"feature_importance.csv",index=False)
    pd.DataFrame(boot).to_csv(METRICS/"student_cluster_bootstrap.csv",index=False)
    print("Permutation importance, masked examples, and student-cluster bootstrap intervals saved.")








