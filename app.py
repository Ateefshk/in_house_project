from pathlib import Path
import sys
import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
from models import EXP  # noqa: E402

METRICS = ROOT / "outputs" / "metrics"
FIGURES = ROOT / "outputs" / "figures"
MODELS = ROOT / "models"
OUTCOMES = ["Distinction", "Pass", "Fail", "Withdrawn"]

st.set_page_config(
    page_title="Student Performance Explorer",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
[data-testid="stMetric"] {background: #f3f7fb; border: 1px solid #dbe6ef; padding: 14px; border-radius: 12px;}
div[data-testid="stTabs"] button {font-size: 1rem;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading fitted prediction models…")
def load_model(task):
    path = MODELS / f"{task}_selected_expanded_day28.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Fitted model not found: {path}")
    return joblib.load(path)

@st.cache_data(show_spinner=False)
def load_csv(path):
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)

def get_probabilities(estimator, row, labels):
    raw = estimator.predict_proba(row[EXP])[0]
    classes = list(estimator.classes_)
    return {label: float(raw[classes.index(label)]) if label in classes else 0.0 for label in labels}

@st.cache_data(show_spinner=False)
def get_course_pairs():
    source = ROOT / "data" / "course_presentations.csv"
    if not source.exists():
        source = ROOT / "data" / "original" / "studentInfo.csv"
    pairs = pd.read_csv(source, usecols=["code_module", "code_presentation"])
    pairs = pairs.drop_duplicates().sort_values(["code_module", "code_presentation"])
    return {
        module: sorted(pairs.loc[pairs.code_module.eq(module), "code_presentation"].unique().tolist())
        for module in pairs.code_module.unique()
    }


def prediction_inputs():
    st.caption("Enter a hypothetical learner profile. The fitted models use the expanded feature set and activity through course day 27 (the first 28 days).")
    c1, c2, c3 = st.columns(3)
    presentations = get_course_pairs()
    modules = list(presentations)
    with c1:
        module = st.selectbox("Course module", modules, index=5)
        presentation = st.selectbox("Presentation", presentations[module])
        registration = st.slider("Registration day relative to course start", -311, 27, -53)
        credits = st.select_slider("Studied credits", options=[30, 60, 90, 120, 150, 180, 240, 300, 360, 420, 480, 540, 600, 630], value=60)
    with c2:
        clicks = st.number_input("VLE clicks in first 28 days", min_value=0, max_value=6084, value=180, step=10)
        active_days = st.slider("Days with at least one VLE click", 0, 28, 10)
        previous_attempts = st.slider("Previous attempts", 0, 6, 0)
        enrolled_days = st.slider("Enrolled days available by day 27", 1, 28, 28)
    with c3:
        assessments_due = st.slider("Assessments due by cutoff", 0, 2, 1)
        assessments_submitted = st.slider("Due assessments submitted by cutoff", 0, assessments_due, min(1, assessments_due))
        assessments_missed = assessments_due - assessments_submitted
        st.metric("Due assessments not submitted", assessments_missed)
        st.caption("The app derives missed assessments from due minus submitted.")
    active_days = min(active_days, enrolled_days)
    clicks_per_day = clicks / max(enrolled_days, 1)
    active_rate = active_days / max(enrolled_days, 1)
    row = pd.DataFrame([{
        "code_module": module,
        "code_presentation": presentation,
        "date_registration": registration,
        "clicks": clicks,
        "active_days": active_days,
        "enrolled_days_by_cutoff": enrolled_days,
        "clicks_per_enrolled_day": clicks_per_day,
        "active_day_rate": active_rate,
        "log_clicks": float(np.log1p(clicks)),
        "num_of_prev_attempts": previous_attempts,
        "studied_credits": credits,
        "assessments_due": assessments_due,
        "assessments_submitted": assessments_submitted,
        "assessments_missed": assessments_missed,
    }])
    return row

st.title("🎓 Student Performance Explorer")
st.write("Explore the trained OULAD models, review their evaluation, or try a prediction with a hypothetical 28-day learner profile.")
st.info("Research demonstration using historical data from one UK university. Predictions are not diagnoses or decisions about real students, and do not establish that interventions improve outcomes.")

prediction_tab, results_tab, data_tab, about_tab = st.tabs([
    "🔮 Try a prediction", "📊 Model results", "📈 Explore the data", "ℹ️ About"
])

with prediction_tab:
    left, right = st.columns([1.05, 0.95], gap="large")
    with left:
        row = prediction_inputs()
        go = st.button("Predict performance", type="primary", use_container_width=True)
    with right:
        threshold = st.slider("At-risk alert threshold", 0.20, 0.60, 0.50, 0.01,
                              help="This threshold changes the displayed alert. It does not retrain the model.")
        st.caption("The default threshold is 0.50. Lower thresholds flag more profiles and can produce more false alarms.")
        if go:
            try:
                binary = load_model("binary")
                multi = load_model("multiclass")
                risk = get_probabilities(binary, row, [0, 1])
                outcome = get_probabilities(multi, row, OUTCOMES)
                risk_probability = risk[1]
                predicted = max(outcome, key=outcome.get)
                a, b = st.columns(2)
                a.metric("At-risk probability", f"{risk_probability:.1%}")
                b.metric("Predicted performance category", predicted)
                if risk_probability >= threshold:
                    st.warning(f"This hypothetical profile is above the selected threshold ({threshold:.0%}).")
                else:
                    st.success(f"This hypothetical profile is below the selected threshold ({threshold:.0%}).")
                st.subheader("Four-category probabilities")
                out_df = pd.DataFrame({"Outcome": OUTCOMES, "Probability": [outcome[x] for x in OUTCOMES]}).set_index("Outcome")
                st.bar_chart(out_df, horizontal=True, color="#42789f")
                st.caption("The binary and four-category models are separate fitted models, so their probabilities summarize different prediction tasks.")
            except (FileNotFoundError, ValueError, AttributeError) as exc:
                st.error(f"Could not load or use the fitted model: {exc}")
        else:
            st.subheader("Your prediction will appear here")
            st.write("Adjust the profile on the left, then select **Predict performance**.")
            st.caption("No profile is stored by this local app.")

with results_tab:
    st.subheader("Held-out test-set performance")
    task_choice = st.radio("Prediction task", ["Binary risk", "Four-category result"], horizontal=True)
    task = "binary" if task_choice == "Binary risk" else "multiclass"
    metric_path = METRICS / f"{task}_selected_test_metrics.csv"
    metrics = load_csv(metric_path)
    if metrics.empty:
        st.warning("Run `python main.py` first to generate evaluation results.")
    else:
        m = metrics.iloc[0]
        cols = st.columns(4)
        cols[0].metric("Accuracy", f"{m['accuracy']:.3f}")
        cols[1].metric("Macro-F1", f"{m['macro_f1']:.3f}")
        cols[2].metric("Balanced accuracy", f"{m['balanced_accuracy']:.3f}")
        if task == "binary":
            cols[3].metric("At-risk recall", f"{m['recall_1']:.3f}")
            cols2 = st.columns(4)
            cols2[0].metric("At-risk precision", f"{m['precision_1']:.3f}")
            cols2[1].metric("ROC-AUC", f"{m['roc_auc']:.3f}")
            cols2[2].metric("Average precision", f"{m['average_precision']:.3f}")
            cols2[3].metric("Log loss", f"{m['log_loss']:.3f}")
            threshold_results = load_csv(METRICS / "threshold_test_evaluation.csv")
            if not threshold_results.empty:
                st.subheader("Threshold trade-off on the untouched test set")
                st.dataframe(threshold_results[[c for c in ["operating_point", "threshold", "precision", "recall", "f1", "false_negatives", "false_positives", "alerts"] if c in threshold_results.columns]], hide_index=True, use_container_width=True)
        else:
            cols[3].metric("Macro OVR ROC-AUC", f"{m['macro_ovr_roc_auc']:.3f}")
            st.metric("Log loss", f"{m['log_loss']:.3f}")
        cm_path = METRICS / f"{task}_confusion_matrix.csv"
        cm = load_csv(cm_path)
        if not cm.empty:
            st.subheader("Confusion matrix")
            st.dataframe(cm, use_container_width=True)
        model_comparison = load_csv(METRICS / f"{task}_model_comparison.csv")
        if not model_comparison.empty:
            expanded = model_comparison[(model_comparison.get("split") == "test") & (model_comparison.get("features") == "expanded")]
            if not expanded.empty:
                st.subheader("Candidate models")
                st.dataframe(expanded[[c for c in ["model", "accuracy", "macro_f1", "balanced_accuracy", "log_loss"] if c in expanded.columns]].sort_values("macro_f1", ascending=False), hide_index=True, use_container_width=True)
    st.caption("Model selection used training-only grouped cross-validation. The test set was held out from model and threshold selection.")

with data_tab:
    st.subheader("Cohort and activity patterns")
    cohorts = load_csv(METRICS / "cohort_eligibility.csv")
    if not cohorts.empty:
        cutoff = st.select_slider("Prediction window", options=[7, 14, 21, 28], value=28, format_func=lambda x: f"First {x} days")
        subset = cohorts[cohorts.cutoff_days == cutoff]
        if not subset.empty:
            r = subset.iloc[0]
            c1, c2, c3 = st.columns(3)
            c1.metric("Eligible enrollments", f"{int(r.eligible_enrollments):,}")
            c2.metric("Distinct students", f"{int(r.eligible_students):,}")
            c3.metric("At-risk prevalence", f"{r.at_risk_prevalence:.1%}")
    figure_options = {
        "Original outcome distribution": "original_cohort_outcomes.png",
        "Eligible day-28 outcome distribution": "eligible_cohort_outcomes.png",
        "Early activity distribution": "engagement_distribution.png",
        "Activity by eventual outcome": "engagement_by_outcome.png",
        "Missing values": "missing_values.png",
        "Temporal versus primary performance": "temporal_vs_primary.png",
    }
    selected_figure = st.selectbox("Figure", list(figure_options))
    image_path = FIGURES / figure_options[selected_figure]
    if image_path.exists():
        st.image(str(image_path), use_container_width=True)
    else:
        st.info("This figure will appear after the project pipeline has been run.")
    st.caption("OULAD is observational. Activity patterns do not show that clicks cause academic outcomes.")

with about_tab:
    st.subheader("What this app does")
    st.markdown("""
- Loads the saved day-28 expanded-feature binary and multiclass pipelines.
- Lets you explore held-out metrics, threshold choices, cohort summaries, and saved figures.
- Generates predictions only for hypothetical profiles entered in this session; it does not save those profiles.

**Important limitations:** OULAD represents one UK university. The predictions are research outputs, not validated intervention decisions. The app does not demonstrate causal effects or improved student outcomes. The two task models are separate and can produce different summaries.
""")
    st.write("Model feature inputs:", ", ".join(EXP))
    st.write("This app uses the expanded feature set and the 28-day fitted models. It does not change the cutoff used by those models.")
    st.write("See `README.md` for methodology and interpretation limits.")


