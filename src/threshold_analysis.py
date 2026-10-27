"""Validation-only threshold analysis; selected threshold is applied once to test."""
import os
os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from common import PREDICTIONS, METRICS, FIGURES

def run_threshold_analysis():
    val=pd.read_csv(PREDICTIONS/"binary_validation_predictions.csv",dtype={"id_student":"string"})
    test=pd.read_csv(PREDICTIONS/"binary_test_predictions.csv",dtype={"id_student":"string"})
    if val.empty or test.empty: raise ValueError("Binary validation/test predictions are missing")
    if val.model.nunique()!=1 or test.model.nunique()!=1: raise ValueError("Expected one selected binary model in prediction files")
    y=val.y_true.astype(int).to_numpy(); p=val.prob_1.to_numpy()
    thresholds=sorted(set([.20,.25,.30,.35,.3779,.40,.45,.50,.60,*np.quantile(p,np.linspace(.05,.95,19)).tolist()]))
    rows=[]
    for th in thresholds:
        pred=(p>=th).astype(int); cm=confusion_matrix(y,pred,labels=[0,1]); prec,rec,f1,_=precision_recall_fscore_support(y,pred,average="binary",zero_division=0)
        tp=int(cm[1,1]); fp=int(cm[0,1]); alerts=int(pred.sum())
        if alerts != tp+fp: raise AssertionError(f"Alert count {alerts} does not equal TP+FP ({tp+fp})")
        rows.append({"threshold":float(th),"precision":prec,"recall":rec,"f1":f1,"false_negatives":int(cm[1,0]),"false_positives":fp,"true_positives":tp,"alerts":alerts,"alert_rate":float(pred.mean())})
    table=pd.DataFrame(rows).drop_duplicates("threshold").sort_values("threshold")
    table.to_csv(METRICS/"threshold_analysis.csv",index=False)
    chosen=table.sort_values(["f1","recall","threshold"],ascending=[False,False,True]).iloc[0]
    # One untouched test evaluation at the validation-selected point and default .50.
    yt=test.y_true.astype(int).to_numpy(); pt=test.prob_1.to_numpy(); results=[]
    for name,th in [("default_0.50",.5),("validation_best_f1",float(chosen.threshold))]:
        yp=(pt>=th).astype(int); cm=confusion_matrix(yt,yp,labels=[0,1]); prec,rec,f1,_=precision_recall_fscore_support(yt,yp,average="binary",zero_division=0)
        alerts=int(yp.sum()); tp=int(cm[1,1]); fp=int(cm[0,1])
        if alerts != tp+fp: raise AssertionError("Test alerts must equal true positives plus false positives")
        results.append({"operating_point":name,"threshold":th,"validation_f1":float(chosen.f1) if name!="default_0.50" else np.nan,"precision":prec,"recall":rec,"f1":f1,"false_negatives":int(cm[1,0]),"false_positives":fp,"true_positives":tp,"alerts":alerts,"alert_rate":float(yp.mean()),"confusion_matrix":str(cm.tolist())})
    pd.DataFrame(results).to_csv(METRICS/"threshold_test_evaluation.csv",index=False)
    fig,ax=plt.subplots(figsize=(6.5,4.8)); ax.plot(table.recall,table.precision,"o-",ms=3); ax.set(xlabel="Validation recall",ylabel="Validation precision",title="Validation alert threshold trade-off"); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(FIGURES/"threshold_precision_recall.png",dpi=220); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4.8)); ax.plot(table.threshold,table.recall,label="Recall"); ax.plot(table.threshold,table.precision,label="Precision"); ax.axvline(chosen.threshold,color="black",linestyle="--",label=f"Validation F1 choice {chosen.threshold:.3f}"); ax.set(xlabel="Threshold",ylabel="Score",title="Threshold versus recall and precision"); ax.legend(frameon=False); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(FIGURES/"threshold_recall_precision.png",dpi=220); plt.close(fig)
    print(f"Validation-selected threshold {chosen.threshold:.4f}; selection based on validation F1 only.")
    return float(chosen.threshold)

if __name__=="__main__": run_threshold_analysis()



