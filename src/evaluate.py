"""Metric calculation, independent reconciliation, and publication-style figures."""
import os
os.environ.setdefault("MPLBACKEND", "Agg")
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, f1_score, balanced_accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report, roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score, log_loss, brier_score_loss)
from sklearn.calibration import calibration_curve
from common import METRICS, PREDICTIONS, FIGURES, OUTCOMES
sns.set_theme(style="whitegrid", context="notebook")

def metric_dict(y_true, y_pred, proba, labels):
    y=np.asarray(y_true); pred=np.asarray(y_pred); proba=np.asarray(proba)
    d={"n":len(y),"accuracy":accuracy_score(y,pred),"macro_f1":f1_score(y,pred,average="macro"),
       "weighted_f1":f1_score(y,pred,average="weighted"),"balanced_accuracy":balanced_accuracy_score(y,pred)}
    pp,rr,ff,_=precision_recall_fscore_support(y,pred,labels=labels,zero_division=0)
    for c,p,r,f in zip(labels,pp,rr,ff): d.update({f"precision_{c}":p,f"recall_{c}":r,f"f1_{c}":f})
    if len(labels)==2:
        d["roc_auc"]=roc_auc_score(y,proba[:,1]); d["average_precision"]=average_precision_score(y,proba[:,1])
        d["brier_score"]=brier_score_loss(y,proba[:,1]); d["log_loss"]=log_loss(y,proba,labels=labels)
    else:
        ordered=sorted(labels); order=[labels.index(x) for x in ordered]; pr=proba[:,order]
        d["macro_ovr_roc_auc"]=roc_auc_score(y,pr,labels=ordered,multi_class="ovr",average="macro")
        d["macro_ovr_average_precision"]=average_precision_score(pd.get_dummies(pd.Categorical(y,categories=ordered)),pr,average="macro")
        d["log_loss"]=log_loss(y,pr,labels=ordered)
        yhot=pd.get_dummies(pd.Categorical(y,categories=labels)).to_numpy()
        d["multiclass_brier"]=float(np.mean(np.sum((proba-yhot)**2,axis=1)))
    d["confusion_matrix"]=json.dumps(confusion_matrix(y,pred,labels=labels).tolist())
    return d

def classification_tables(y, pred, labels, task_name, stem):
    cm=confusion_matrix(y,pred,labels=labels)
    pd.DataFrame(cm,index=[f"actual_{x}" for x in labels],columns=[f"pred_{x}" for x in labels]).to_csv(METRICS/f"{stem}_confusion_matrix.csv")
    pd.DataFrame(classification_report(y,pred,labels=labels,output_dict=True,zero_division=0)).T.to_csv(METRICS/f"{stem}_classification_report.csv")

def plot_confusion(y, pred, labels, title, path):
    fig,ax=plt.subplots(figsize=(6,5)); sns.heatmap(confusion_matrix(y,pred,labels=labels),annot=True,fmt="d",cmap="Blues",xticklabels=labels,yticklabels=labels,ax=ax)
    ax.set(xlabel="Predicted",ylabel="Actual",title=title); fig.tight_layout(); fig.savefig(path,dpi=220,bbox_inches="tight"); plt.close(fig)

def plot_curves(y, proba, labels, stem):
    if len(labels)==2:
        fpr,tpr,_=roc_curve(y,proba[:,1]); fig,ax=plt.subplots(figsize=(5.6,4.6)); ax.plot(fpr,tpr,label=f"AUC {roc_auc_score(y,proba[:,1]):.3f}"); ax.plot([0,1],[0,1],"--",color="gray"); ax.set(xlabel="False positive rate",ylabel="True positive rate",title="Binary ROC curve"); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(FIGURES/f"{stem}_roc.png",dpi=220); plt.close(fig)
        pr,rc,_=precision_recall_curve(y,proba[:,1]); fig,ax=plt.subplots(figsize=(5.6,4.6)); ax.plot(rc,pr,label=f"AP {average_precision_score(y,proba[:,1]):.3f}"); ax.set(xlabel="Recall",ylabel="Precision",title="Binary precision-recall curve"); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(FIGURES/f"{stem}_precision_recall.png",dpi=220); plt.close(fig)
        frac,mean=calibration_curve(y,proba[:,1],n_bins=10,strategy="quantile"); fig,ax=plt.subplots(figsize=(5.6,4.6)); ax.plot(mean,frac,"o-",label="Model"); ax.plot([0,1],[0,1],"--",color="gray",label="Perfect calibration"); ax.set(xlabel="Mean predicted risk",ylabel="Observed risk",title="Binary calibration curve"); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(FIGURES/f"{stem}_calibration.png",dpi=220); plt.close(fig)
    else:
        ordered=sorted(labels); ix=[labels.index(x) for x in ordered]; yhot=pd.get_dummies(pd.Categorical(y,categories=ordered)).to_numpy()
        fig,ax=plt.subplots(figsize=(6.2,5.2))
        for j,c in enumerate(ordered):
            fpr,tpr,_=roc_curve(yhot[:,j],proba[:,ix[j]]); ax.plot(fpr,tpr,label=c)
        ax.plot([0,1],[0,1],"--",color="gray"); ax.set(xlabel="False positive rate",ylabel="True positive rate",title="Four-category one-vs-rest ROC"); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(FIGURES/f"{stem}_roc.png",dpi=220); plt.close(fig)

def run_evaluation():
    """Recompute the selected model's test metrics from saved probabilities."""
    for task,labels in [("binary",[0,1]),("multiclass",OUTCOMES)]:
        pred_path=PREDICTIONS/f"{task}_test_predictions.csv"
        if not pred_path.exists(): continue
        df=pd.read_csv(pred_path,dtype={"id_student":"string"})
        y=df.y_true.to_numpy(dtype=int if task=="binary" else str); yp=df.y_pred.to_numpy(dtype=int if task=="binary" else str)
        prob=df[[f"prob_{x}" for x in labels]].to_numpy()
        m=metric_dict(y,yp,prob,labels); m["task"]=task; m["model"]=df.model.iloc[0]
        pd.DataFrame([m]).to_csv(METRICS/f"{task}_selected_test_metrics.csv",index=False)
        classification_tables(y,yp,labels,task,task)
        plot_confusion(y,yp,labels,f"{task.title()} test confusion matrix",FIGURES/f"{task}_confusion_matrix.png")
        plot_curves(y,prob,labels,task)
    subgroup_evaluation()
    print("Selected-model metrics and confusion/ROC/PR/calibration figures saved from test predictions.")

if __name__ == "__main__": run_evaluation()

def plot_dataset_diagnostics(features):
    """Create cohort, missingness, and engagement/outcome figures without altering data."""
    from common import FIGURES, SOURCE, METRICS, OUTCOMES
    import pandas as pd
    import matplotlib.pyplot as plt
    base=pd.read_csv(SOURCE/"studentInfo.csv")
    fig,ax=plt.subplots(figsize=(7,4.5)); base.final_result.value_counts().reindex(OUTCOMES).plot.bar(ax=ax,color="#7895b2"); ax.set(title="Original cohort outcome distribution",xlabel="Final result",ylabel="Enrollments"); fig.tight_layout(); fig.savefig(FIGURES/"original_cohort_outcomes.png",dpi=220); plt.close(fig)
    elig=features[features.cutoff_days.eq(28)]
    fig,ax=plt.subplots(figsize=(7,4.5)); elig.final_result.value_counts().reindex(OUTCOMES).plot.bar(ax=ax,color="#507d6a"); ax.set(title="Eligible 28-day cohort outcomes",xlabel="Final result",ylabel="Enrollments"); fig.tight_layout(); fig.savefig(FIGURES/"eligible_cohort_outcomes.png",dpi=220); plt.close(fig)
    miss=pd.read_csv(METRICS/"missing_values.csv"); miss=miss[miss.missing_n>0].sort_values("missing_rate",ascending=False).head(20)
    fig,ax=plt.subplots(figsize=(8,5)); ax.barh(miss.file+" · "+miss.column,miss.missing_rate,color="#c17b52"); ax.invert_yaxis(); ax.set(xlabel="Missing-value rate",title="Fields with the most missing values"); fig.tight_layout(); fig.savefig(FIGURES/"missing_values.png",dpi=220); plt.close(fig)
    day=features[features.cutoff_days.eq(28)]
    fig,ax=plt.subplots(figsize=(7,4.5)); ax.hist(np.log1p(day.clicks),bins=35,color="#6987a6",alpha=.85); ax.set(xlabel="log(1 + clicks), days 0–27",ylabel="Enrollments",title="Early engagement distribution"); fig.tight_layout(); fig.savefig(FIGURES/"engagement_distribution.png",dpi=220); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.8)); day.boxplot(column="active_day_rate",by="final_result",ax=ax,grid=False,showfliers=False); ax.set(title="Active-day rate by eventual outcome",xlabel="Final result",ylabel="Active days / enrolled days"); fig.suptitle(""); fig.tight_layout(); fig.savefig(FIGURES/"engagement_by_outcome.png",dpi=220); plt.close(fig)

def subgroup_evaluation():
    """Describe binary model errors by module/presentation; do not infer fairness from small cells."""
    from common import PREDICTIONS, OUT, KEY
    pred=pd.read_csv(PREDICTIONS/"binary_test_predictions.csv",dtype={"id_student":"string"})
    feat=pd.read_csv(OUT/"features_by_cutoff.csv",dtype={"id_student":"string"})
    feat=feat[feat.cutoff_days.eq(28)][KEY+["final_result"]]
    df=pred.merge(feat,on=KEY,how="left",validate="one_to_one"); df.y_true=df.y_true.astype(int); df.y_pred=df.y_pred.astype(int)
    rows=[]
    for dim in ["code_module","code_presentation"]:
        for name,g in df.groupby(dim):
            cm=confusion_matrix(g.y_true,g.y_pred,labels=[0,1]); precision,recall,f1,_=precision_recall_fscore_support(g.y_true,g.y_pred,average="binary",zero_division=0)
            rows.append({"grouping":dim,"group":name,"n":len(g),"students":g.id_student.nunique(),"at_risk_prevalence":float(g.y_true.mean()),"precision":precision,"recall":recall,"f1":f1,"false_positive_rate":float(cm[0,1]/max(1,cm[0].sum())),"false_negatives":int(cm[1,0]),"false_positives":int(cm[0,1]),"small_cell_caution":len(g)<100})
    pd.DataFrame(rows).to_csv(METRICS/"binary_test_subgroup_metrics.csv",index=False)



if __name__ == '__main__':
    run_evaluation()




