"""Single command to inspect data, build features, train, validate, explain, and audit."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
import pandas as pd
from common import ROOT, METRICS, OUT, KEY, ensure_dirs
from inspect_data import run_inspection
from feature_engineering import build_features
from preprocess import assign_group_splits
from train_binary import run_binary
from train_multiclass import run_multiclass
from threshold_analysis import run_threshold_analysis
from temporal_validation import run_temporal_validation
from explainability import run_explainability
from evaluate import run_evaluation, plot_dataset_diagnostics, metric_dict

def independent_checks(assignments):
    rows=[]
    rows.append({"check":"unique student partition","pass":assignments.groupby("id_student").split.nunique().max()==1,"value":int(assignments.groupby("id_student").split.nunique().max())})
    for task,labels in [("binary",[0,1]),("multiclass",["Distinction","Pass","Fail","Withdrawn"])]:
        df=pd.read_csv(OUT/"predictions"/f"{task}_test_predictions.csv",dtype={"id_student":"string"})
        y=df.y_true.to_numpy(dtype=int if task=="binary" else str); pred=df.y_pred.to_numpy(dtype=int if task=="binary" else str); prob=df[[f"prob_{x}" for x in labels]].to_numpy()
        rows.append({"check":f"{task} probability rows sum to one","pass":bool(abs(prob.sum(axis=1)-1).max()<1e-10),"value":len(df)})
        actual=metric_dict(y,pred,prob,labels)
        saved=pd.read_csv(METRICS/f"{task}_selected_test_metrics.csv").iloc[0]
        for metric in ["accuracy","macro_f1","weighted_f1","balanced_accuracy","log_loss"]:
            rows.append({"check":f"{task} saved/recomputed {metric}","pass":bool(abs(float(saved[metric])-float(actual[metric]))<1e-12),"value":len(df)})
        rows.append({"check":f"{task} confusion matrix reconciles","pass":str(saved.confusion_matrix)==str(actual["confusion_matrix"]),"value":len(df)})
    q=pd.DataFrame(rows); q.to_csv(METRICS/"verification_summary.csv",index=False)
    if not q["pass"].all(): raise AssertionError(f"Verification checks failed:\n{q.loc[~q['pass']].to_string(index=False)}")
    print(f"Independent verification passed ({len(q)} checks).")

def main():
    ensure_dirs()
    print("Stage 1/8: inspect source files and report quality..."); run_inspection()
    print("Stage 2/8: reconstruct actual cutoff features..."); features=build_features()
    print("Stage 3/8: build one stratified student-group split shared by both tasks...")
    basis=features[KEY+["final_result","at_risk"]].drop_duplicates(KEY)
    assignments=assign_group_splits(basis,"final_result")
    assignments.to_csv(METRICS/"student_split_assignments.csv",index=False)
    print("Stage 4/8: binary model selection and evaluation..."); run_binary(features,assignments)
    print("Stage 5/8: four-category model selection and evaluation..."); run_multiclass(features,assignments)
    print("Stage 6/8: validation thresholding, metrics, and figures..."); run_threshold_analysis(); run_evaluation(); plot_dataset_diagnostics(features)
    print("Stage 7/8: temporal holdout and explainability..."); run_temporal_validation(features); run_explainability()
    print("Stage 8/8: verify saved predictions, probabilities, and student grouping..."); independent_checks(assignments)
    print(f"Finished. Outputs are in {OUT}")

if __name__=="__main__":
    try: main()
    except (FileNotFoundError,ValueError,KeyError,AssertionError) as exc:
        raise SystemExit(f"Pipeline stopped with a validated input or analysis error: {exc}")


