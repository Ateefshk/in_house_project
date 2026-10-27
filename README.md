# Improved OULAD academic performance project

A reproducible student-grouped study for early prediction of OULAD final outcomes. This repository contains analysis code, fitted day-28 models, aggregate evaluation outputs, and an interactive dashboard. Raw student-level source data, row-level predictions, and the accompanying research paper are excluded. The analysis reconstructs actual day 0–6, 0–13, 0–20, and 0–27 VLE activity from `studentVle.csv`; it never creates artificial dates. Both targets are supported: binary risk (Fail/Withdrawn versus Pass/Distinction) and the original four-category result.

## Run the interactive dashboard

Use Python 3.13.2, install the dependencies, then launch Streamlit from this folder:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The dashboard works with the included fitted models and aggregate results. See [INTERACTIVE_APP.md](INTERACTIVE_APP.md) for details.

## Reproduce model training

Obtain the official OULAD CSV archive from [UCI](https://archive.ics.uci.edu/dataset/349/open%2Buniversity%2B), place its CSV files in `data/original/`, and run:

```powershell
python main.py
```

See [data/README.md](data/README.md) for attribution and placement details. The training pipeline writes generated metrics, predictions, and figures under `outputs/`; fitted pipelines are saved in `models/`. Student-level predictions and split assignments are generated locally and must not be published.

## Method

- Source inspection validates schemas, row counts, data types, missing values, duplicates, enrollment keys, outcomes, and cutoff eligibility.
- Eligibility is recalculated separately for each cutoff. A known registration day must be on or before the cutoff; a withdrawal on/before the cutoff is ineligible; a withdrawn record with unknown unregistration timing is excluded as ambiguous.
- Predictors use module, presentation, registration timing, cutoff-limited clicks and active days, enrolled days, engagement rates, log clicks, prior attempts, studied credits, and assessment due/submission participation observed by cutoff. Assessment scores are excluded because score publication dates are unavailable. Final result and unregistration are never predictors. Student ID is only used as a group key.
- `StratifiedGroupKFold` creates approximate 70/15/15 partitions grouped by student and stratified by final result.
- Candidate models are evaluated with training-only grouped CV. The selected model is evaluated against validation and untouched test data. The binary threshold is selected on validation. A separate temporal evaluation uses earlier presentations and tests on 2014J.
- OULAD represents one UK university. Results do not establish generalization to other institutions, causal effects, intervention effectiveness, or fair treatment of protected groups. Predictions are for research demonstration and require human review.

## Project contents

`src/` and `main.py` contain the training pipeline; `app.py` is the interactive dashboard; `models/` contains fitted day-28 pipelines; `outputs/metrics/` and `outputs/figures/` contain aggregate evaluation artifacts. Raw data and the IEEE research paper are not part of this project repository.
