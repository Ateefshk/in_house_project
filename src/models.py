"""Scikit-learn model factories, including group-aware soft voting and stacking."""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GroupKFold
from common import SEED

CAT=["code_module","code_presentation"]
BASE=["code_module","code_presentation","date_registration","clicks","active_days","enrolled_days_by_cutoff","clicks_per_enrolled_day","active_day_rate","log_clicks"]
EXP=BASE+["num_of_prev_attempts","studied_credits","assessments_due","assessments_submitted","assessments_missed"]

def make_preprocessor(features):
    cats=[c for c in features if c in CAT]; nums=[c for c in features if c not in cats]
    return ColumnTransformer([("num",Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler())]),nums),
        ("cat",Pipeline([("impute",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]),cats)])

def simple_pipe(model,features): return Pipeline([("preprocess",make_preprocessor(features)),("model",model)])

def candidate_names():
    names=["Dummy","LogisticRegression","DecisionTree","RandomForest","GradientBoosting","SoftVoting","Stacking"]
    try:
        import xgboost  # noqa: F401
        names.append("XGBoost")
    except ImportError: pass
    try:
        import lightgbm  # noqa: F401
        names.append("LightGBM")
    except ImportError: pass
    return names

def make_candidate(name,task,features,groups=None):
    binary=task=="binary"; cw="balanced" if binary else None
    lr=LogisticRegression(max_iter=1500,C=.5,class_weight=cw,random_state=SEED)
    rf=RandomForestClassifier(n_estimators=120,min_samples_leaf=5,max_features="sqrt",class_weight="balanced_subsample" if binary else None,n_jobs=-1,random_state=SEED)
    dt=DecisionTreeClassifier(max_depth=8,min_samples_leaf=20,class_weight=cw,random_state=SEED)
    if name=="Dummy": return simple_pipe(DummyClassifier(strategy="prior"),features)
    if name=="LogisticRegression": return simple_pipe(lr,features)
    if name=="DecisionTree": return simple_pipe(dt,features)
    if name=="RandomForest": return simple_pipe(rf,features)
    if name=="GradientBoosting": return simple_pipe(GradientBoostingClassifier(n_estimators=90,max_depth=2,learning_rate=.05,random_state=SEED),features)
    base=[("lr",simple_pipe(lr,features)),("rf",simple_pipe(rf,features))]
    if name=="SoftVoting": return VotingClassifier(estimators=base,voting="soft",n_jobs=1)
    if name=="Stacking":
        if groups is None: raise ValueError("Stacking requires student IDs for group-aware inner folds")
        inner=list(GroupKFold(n_splits=3).split(np.zeros(len(groups)),np.zeros(len(groups)),groups))
        return StackingClassifier(estimators=base,final_estimator=LogisticRegression(max_iter=1500,class_weight=cw),cv=inner,stack_method="predict_proba",n_jobs=1,passthrough=False)
    # Optional libraries are added only when importable and versions are compatible.
    if name=="XGBoost":
        from xgboost import XGBClassifier
        return simple_pipe(XGBClassifier(n_estimators=120,max_depth=4,learning_rate=.05,subsample=.85,colsample_bytree=.85,eval_metric="logloss",random_state=SEED),features)
    if name=="LightGBM":
        from lightgbm import LGBMClassifier
        return simple_pipe(LGBMClassifier(n_estimators=120,learning_rate=.05,random_state=SEED,verbosity=-1),features)
    raise ValueError(f"Unknown estimator {name}")

