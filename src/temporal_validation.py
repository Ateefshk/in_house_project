"""Earlier-presentation model selection and untouched 2014J temporal evaluation."""
import os
os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_fscore_support
from common import KEY, OUT, METRICS, PREDICTIONS, FIGURES, OUTCOMES
from preprocess import assign_group_splits
from training import candidates
from models import EXP, make_candidate
from evaluate import metric_dict, plot_confusion

def select_on_earlier_train(train,task):
    target="at_risk" if task=="binary" else "final_result"; y=train[target].to_numpy(dtype=int if task=="binary" else str); groups=train.id_student.astype("string").to_numpy(); X=train[EXP]
    cv=GroupKFold(n_splits=3); rows=[]
    for name in candidates():
        scores=[]
        for a,b in cv.split(X,y,groups):
            model=make_candidate(name,task,EXP,groups[a] if name=="Stacking" else None); model.fit(X.iloc[a],y[a]); pred=model.predict(X.iloc[b])
            score=f1_score(y[b],pred,average="binary",pos_label=1,zero_division=0) if task=="binary" else f1_score(y[b],pred,average="macro",zero_division=0)
            scores.append(score)
        rows.append({"model":name,"grouped_cv_mean":float(np.mean(scores)),"grouped_cv_sd":float(np.std(scores,ddof=1))})
    table=pd.DataFrame(rows).sort_values("grouped_cv_mean",ascending=False); table.to_csv(METRICS/f"temporal_{task}_model_selection_cv.csv",index=False)
    return table.iloc[0].model

def run_temporal_validation(features):
    d=features[features.cutoff_days.eq(28)].copy()
    test=d[d.code_presentation.eq("2014J")].copy()
    if test.empty: raise ValueError("No eligible day-28 2014J records are available")
    overlap=set(test.id_student.astype(str))
    earlier=d[~d.code_presentation.eq("2014J")].copy()
    earlier=earlier[~earlier.id_student.astype(str).isin(overlap)].copy()
    if earlier.empty: raise ValueError("No earlier training records remain after removing overlapping students")
    rows=[]; all_predictions=[]
    for task in ["binary","multiclass"]:
        target="at_risk" if task=="binary" else "final_result"; labels=[0,1] if task=="binary" else OUTCOMES
        assignments=assign_group_splits(earlier,target)
        prior=earlier.merge(assignments[KEY+["split"]],on=KEY,validate="one_to_one")
        dev=prior[prior.split.eq("train")]; val=prior[prior.split.eq("validation")]
        if dev.id_student.nunique()<3 or val.empty: raise ValueError("Temporal development split is too small for grouped CV/validation")
        selected=select_on_earlier_train(dev,task)
        val_model=make_candidate(selected,task,EXP,dev.id_student.to_numpy() if selected=="Stacking" else None)
        val_model.fit(dev[EXP],dev[target]); yv=val[target].to_numpy(dtype=int if task=="binary" else str)
        cls=list(val_model.classes_) if hasattr(val_model,"classes_") else list(val_model.named_steps["model"].classes_)
        raw=val_model.predict_proba(val[EXP]); pv=np.column_stack([raw[:,cls.index(c)] if c in cls else np.zeros(len(val)) for c in labels])
        selected_threshold=.5
        if task=="binary":
            ths=sorted(set([.2,.25,.3,.35,.3779,.4,.45,.5,.6,*np.quantile(pv[:,1],np.linspace(.05,.95,19)).tolist()]))
            vals=[]
            for th in ths:
                yp=(pv[:,1]>=th).astype(int); f=f1_score(yv,yp,zero_division=0); vals.append((f,th))
            selected_threshold=sorted(vals,key=lambda z:(z[0],-z[1]),reverse=True)[0][1]
            pd.DataFrame([{"validation_threshold":th,"validation_f1":f} for f,th in vals]).to_csv(METRICS/"temporal_binary_threshold_validation.csv",index=False)
        # Refit only on pre-2014J records after model/threshold decisions are complete.
        final=make_candidate(selected,task,EXP,earlier.id_student.to_numpy() if selected=="Stacking" else None)
        final.fit(earlier[EXP],earlier[target]); yt=test[target].to_numpy(dtype=int if task=="binary" else str)
        ct=list(final.classes_) if hasattr(final,"classes_") else list(final.named_steps["model"].classes_)
        raw=final.predict_proba(test[EXP]); pt=np.column_stack([raw[:,ct.index(c)] if c in ct else np.zeros(len(test)) for c in labels])
        yp=np.asarray(labels)[pt.argmax(axis=1)]
        if task=="binary": yp=(pt[:,1]>=.5).astype(int)
        m=metric_dict(yt,yp,pt,labels); m.update({"task":task,"model":selected,"split":"temporal_2014J_test","threshold":.5 if task=="binary" else np.nan,"train_rows":len(earlier),"train_students":earlier.id_student.nunique(),"test_rows":len(test),"test_students":test.id_student.nunique(),"removed_overlapping_students":len(overlap)})
        rows.append(m)
        if task=="binary":
            yp_selected=(pt[:,1]>=selected_threshold).astype(int); cm=confusion_matrix(yt,yp_selected,labels=[0,1]); pr,rc,f1,_=precision_recall_fscore_support(yt,yp_selected,average="binary",zero_division=0)
            rows.append({"task":task,"model":selected,"split":"temporal_2014J_test_validation_threshold","threshold":selected_threshold,"precision_1":pr,"recall_1":rc,"f1_1":f1,"false_negatives":int(cm[1,0]),"false_positives":int(cm[0,1]),"alerts":int(yp_selected.sum()),"confusion_matrix":str(cm.tolist()),"train_rows":len(earlier),"test_rows":len(test),"removed_overlapping_students":len(overlap)})
        plot_confusion(yt,yp,labels,f"2014J temporal {task} confusion matrix",FIGURES/f"temporal_2014J_{task}_confusion.png")
        for i,(_,r) in enumerate(test.iterrows()):
            all_predictions.append({**{c:r[c] for c in KEY},"task":task,"model":selected,"split":"temporal_2014J_test","y_true":yt[i],"y_pred":yp[i],**{f"prob_{c}":float(pt[i,j]) for j,c in enumerate(labels)},**({"validation_selected_threshold":selected_threshold} if task=="binary" else {})})
    pd.DataFrame(rows).to_csv(METRICS/"temporal_2014J_metrics.csv",index=False)
    pd.DataFrame(all_predictions).to_csv(PREDICTIONS/"temporal_2014J_predictions.csv",index=False)
    # Same-university temporal versus random-holdout display. This is descriptive, not external validation.
    comparison=[]
    for task,metric in [("binary","recall_1"),("multiclass","macro_f1")]:
        primary=pd.read_csv(METRICS/f"{task}_selected_test_metrics.csv").iloc[0]
        temporal=next(r for r in rows if r["task"]==task and r["split"]=="temporal_2014J_test")
        comparison.append({"task":task,"metric":metric,"primary_random_test":float(primary[metric]),"temporal_2014J_test":float(temporal[metric]),"interpretation":"same-university temporal validation; not external validation"})
    comp=pd.DataFrame(comparison); comp.to_csv(METRICS/"temporal_vs_primary_comparison.csv",index=False)
    fig,ax=plt.subplots(figsize=(6.5,4.5)); x=np.arange(len(comp)); w=.35; ax.bar(x-w/2,comp.primary_random_test,w,label="Primary student holdout"); ax.bar(x+w/2,comp.temporal_2014J_test,w,label="Later 2014J presentation"); ax.set_xticks(x,comp.task); ax.set_ylabel("Metric value"); ax.set_title("Primary and temporal performance"); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(FIGURES/"temporal_vs_primary.png",dpi=220); plt.close(fig)
    print(f"2014J temporal validation complete; removed {len(overlap)} overlapping student IDs.")

if __name__=="__main__":
    from feature_engineering import build_features
    run_temporal_validation(build_features())



