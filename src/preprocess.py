"""Student-disjoint splits using stratified grouped folds, with class distribution reports."""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from common import KEY, SEED, METRICS

def assign_group_splits(data, target, n_splits=20):
    """Approximate 70/15/15 using folds; all records for a student share one split."""
    groups = data.id_student.astype("string").to_numpy()
    y = data[target].to_numpy()
    if pd.Series(groups).nunique() < n_splits:
        raise ValueError(f"Need at least {n_splits} unique students for grouped split; found {pd.Series(groups).nunique()}")
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    fold = np.full(len(data), -1, dtype=int)
    for f, (_, holdout) in enumerate(splitter.split(np.zeros(len(data)), y, groups)):
        fold[holdout] = f
    if (fold < 0).any():
        raise RuntimeError("Grouped split did not assign every record")
    split = np.where(fold < 14, "train", np.where(fold < 17, "validation", "test"))
    out = data[KEY].copy()
    out["split"] = split
    out["group_fold"] = fold
    if out.groupby("id_student").split.nunique().max() != 1:
        raise RuntimeError("A student appears in more than one partition")
    return out

def save_split_report(data, assignment, target, task_name):
    q = data[KEY + [target]].merge(assignment[KEY + ["split"]], on=KEY, validate="one_to_one")
    rows=[]
    for split, g in q.groupby("split"):
        dist=g[target].value_counts(normalize=True)
        row={"task":task_name,"split":split,"rows":len(g),"students":g.id_student.nunique()}
        row.update({f"share_{str(k)}":float(v) for k,v in dist.items()})
        rows.append(row)
    pd.DataFrame(rows).to_csv(METRICS / f"{task_name}_split_class_distribution.csv", index=False)
