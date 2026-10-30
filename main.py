"""Interactive predictor and training pipeline for OULAD student outcomes."""
import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR / "src"))
from models import EXP  # noqa: E402

MODELS = ROOT_DIR / "models"
COURSES = ROOT_DIR / "data" / "course_presentations.csv"
OUTCOMES = ["Distinction", "Pass", "Fail", "Withdrawn"]
CREDITS = [30, 60, 90, 120, 150, 180, 240, 300, 360, 420, 480, 540, 600, 630]

def read_course_pairs():
    if not COURSES.exists():
        raise FileNotFoundError(f"Course lookup not found: {COURSES}")
    pairs = pd.read_csv(COURSES, dtype=str)
    return {
        module: sorted(group["code_presentation"].unique().tolist())
        for module, group in pairs.groupby("code_module", sort=True)
    }


def ask_int(label, low, high, default):
    while True:
        raw = input(f"{label} ({low}-{high}) [{default}]: ").strip()
        if not raw:
            return default
        try:
            value = int(raw)
        except ValueError:
            print("Enter a whole number.")
            continue
        if low <= value <= high:
            return value
        print(f"Enter a value from {low} to {high}.")


def ask_choice(label, choices, default=None):
    choices = list(choices)
    while True:
        suffix = f" [{default}]" if default is not None else ""
        value = input(f"{label} ({'/'.join(choices)}){suffix}: ").strip()
        value = value or (default if default is not None else "")
        match = next((item for item in choices if item.casefold() == value.casefold()), None)
        if match is not None:
            return match
        print(f"Choose one of: {', '.join(choices)}.")


def ask_threshold():
    while True:
        raw = input("At-risk alert threshold, 0.20-0.60 [0.50]: ").strip()
        if not raw:
            return 0.50
        try:
            value = float(raw)
        except ValueError:
            print("Enter a number such as 0.50.")
            continue
        if 0.20 <= value <= 0.60:
            return value
        print("Enter a threshold from 0.20 to 0.60.")


def make_profile(presentations):
    print("\nEnter a hypothetical learner profile. Press Enter to use each default.")
    modules = list(presentations)
    module = ask_choice("Course module", modules, default=modules[0])
    presentation = ask_choice("Presentation", presentations[module], default=presentations[module][0])
    registration = ask_int("Registration day relative to course start", -311, 27, -53)
    credits = ask_choice("Studied credits", [str(v) for v in CREDITS], default="60")
    clicks = ask_int("VLE clicks in the first 28 days", 0, 6084, 180)
    enrolled_days = ask_int("Enrolled days available by day 27", 1, 28, 28)
    active_days = ask_int("Days with at least one VLE click", 0, enrolled_days, min(10, enrolled_days))
    previous_attempts = ask_int("Previous attempts", 0, 6, 0)
    assessments_due = ask_int("Assessments due by cutoff", 0, 2, 1)
    assessments_submitted = ask_int("Due assessments submitted by cutoff", 0, assessments_due, min(1, assessments_due))
    return pd.DataFrame([{
        "code_module": module,
        "code_presentation": presentation,
        "date_registration": registration,
        "clicks": clicks,
        "active_days": active_days,
        "enrolled_days_by_cutoff": enrolled_days,
        "clicks_per_enrolled_day": clicks / enrolled_days,
        "active_day_rate": active_days / enrolled_days,
        "log_clicks": float(np.log1p(clicks)),
        "num_of_prev_attempts": previous_attempts,
        "studied_credits": int(credits),
        "assessments_due": assessments_due,
        "assessments_submitted": assessments_submitted,
        "assessments_missed": assessments_due - assessments_submitted,
    }])


def probabilities(estimator, row, labels):
    scores = estimator.predict_proba(row[EXP])[0]
    classes = list(estimator.classes_)
    return {
        label: float(scores[classes.index(label)]) if label in classes else 0.0
        for label in labels
    }


def load_models():
    binary_path = MODELS / "binary_selected_expanded_day28.joblib"
    multiclass_path = MODELS / "multiclass_selected_expanded_day28.joblib"
    for path in (binary_path, multiclass_path):
        if not path.exists():
            raise FileNotFoundError(f"Fitted model not found: {path}")
    print("Loading fitted models...")
    return joblib.load(binary_path), joblib.load(multiclass_path)


def show_prediction(row, binary_model, multiclass_model):
    threshold = ask_threshold()
    risk_prob = probabilities(binary_model, row, [0, 1])[1]
    outcome_prob = probabilities(multiclass_model, row, OUTCOMES)
    predicted = max(outcome_prob, key=outcome_prob.get)

    print("\nPrediction for this hypothetical profile")
    print("-" * 43)
    print(f"Predicted performance category: {predicted}")
    print(f"At-risk probability (Fail or Withdrawn): {risk_prob:.1%}")
    alert = "ABOVE" if risk_prob >= threshold else "BELOW"
    print(f"At-risk alert: {alert} the {threshold:.0%} threshold")
    print("\nFour-category probabilities:")
    for outcome, probability in sorted(outcome_prob.items(), key=lambda item: item[1], reverse=True):
        print(f"  {outcome:<12} {probability:>6.1%}")


def run_predictor():
    try:
        presentations = read_course_pairs()
        binary_model, multiclass_model = load_models()
    except (FileNotFoundError, OSError, ValueError) as exc:
        print(f"Cannot start predictor: {exc}")
        return 1

    print("\nStudent Performance Predictor — first 28 course days")
    print("Type Ctrl+C to exit. No profile is saved.\n")
    try:
        while True:
            row = make_profile(presentations)
            try:
                show_prediction(row, binary_model, multiclass_model)
            except (ValueError, AttributeError) as exc:
                print(f"Could not produce a prediction: {exc}")
            again = input("\nPredict another hypothetical profile? [y/N]: ").strip().casefold()
            if again not in {"y", "yes"}:
                break
    except (KeyboardInterrupt, EOFError):
        print("\nExiting predictor.")
    return 0


def independent_checks(assignments, metrics_dir, output_dir, metric_dict):
    rows = []
    rows.append({"check": "unique student partition", "pass": assignments.groupby("id_student").split.nunique().max() == 1, "value": int(assignments.groupby("id_student").split.nunique().max())})
    for task, labels in [("binary", [0, 1]), ("multiclass", OUTCOMES)]:
        df = pd.read_csv(output_dir / "predictions" / f"{task}_test_predictions.csv", dtype={"id_student": "string"})
        y = df.y_true.to_numpy(dtype=int if task == "binary" else str)
        pred = df.y_pred.to_numpy(dtype=int if task == "binary" else str)
        prob = df[[f"prob_{x}" for x in labels]].to_numpy()
        rows.append({"check": f"{task} probability rows sum to one", "pass": bool(abs(prob.sum(axis=1) - 1).max() < 1e-10), "value": len(df)})
        actual = metric_dict(y, pred, prob, labels)
        saved = pd.read_csv(metrics_dir / f"{task}_selected_test_metrics.csv").iloc[0]
        for metric in ["accuracy", "macro_f1", "weighted_f1", "balanced_accuracy", "log_loss"]:
            rows.append({"check": f"{task} saved/recomputed {metric}", "pass": bool(abs(float(saved[metric]) - float(actual[metric])) < 1e-12), "value": len(df)})
        rows.append({"check": f"{task} confusion matrix reconciles", "pass": str(saved.confusion_matrix) == str(actual["confusion_matrix"]), "value": len(df)})
    checks = pd.DataFrame(rows)
    checks.to_csv(metrics_dir / "verification_summary.csv", index=False)
    if not checks["pass"].all():
        raise AssertionError(f"Verification checks failed:\n{checks.loc[~checks['pass']].to_string(index=False)}")
    print(f"Independent verification passed ({len(checks)} checks).")


def run_training_pipeline():
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

    ensure_dirs()
    print("Stage 1/8: inspect source files and report quality...")
    run_inspection()
    print("Stage 2/8: reconstruct actual cutoff features...")
    features = build_features()
    print("Stage 3/8: build one stratified student-group split shared by both tasks...")
    basis = features[KEY + ["final_result", "at_risk"]].drop_duplicates(KEY)
    assignments = assign_group_splits(basis, "final_result")
    assignments.to_csv(METRICS / "student_split_assignments.csv", index=False)
    print("Stage 4/8: binary model selection and evaluation...")
    run_binary(features, assignments)
    print("Stage 5/8: four-category model selection and evaluation...")
    run_multiclass(features, assignments)
    print("Stage 6/8: validation thresholding, metrics, and figures...")
    run_threshold_analysis()
    run_evaluation()
    plot_dataset_diagnostics(features)
    print("Stage 7/8: temporal holdout and explainability...")
    run_temporal_validation(features)
    run_explainability()
    print("Stage 8/8: verify saved predictions, probabilities, and student grouping...")
    independent_checks(assignments, METRICS, OUT, metric_dict)
    print(f"Finished. Outputs are in {OUT}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Predict a hypothetical OULAD learner profile or run the full training pipeline.")
    parser.add_argument("mode", nargs="?", choices=["predict", "train"], default="predict",
                        help="predict (default) starts the console model; train runs the full pipeline")
    args = parser.parse_args(argv)
    if args.mode == "train":
        try:
            run_training_pipeline()
        except (FileNotFoundError, ValueError, KeyError, AssertionError) as exc:
            raise SystemExit(f"Pipeline stopped with a validated input or analysis error: {exc}")
        return 0
    return run_predictor()


if __name__ == "__main__":
    raise SystemExit(main())
