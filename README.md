# Improved OULAD academic performance project

A reproducible student-grouped study for early prediction of OULAD final outcomes. This repository contains the training pipeline, fitted day-28 models, aggregate evaluation outputs, and an interactive terminal predictor. Raw student-level source data, row-level predictions, and the accompanying research paper are excluded. The analysis reconstructs actual day 0–6, 0–13, 0–20, and 0–27 VLE activity from `studentVle.csv`; it never creates artificial dates. Both targets are supported: binary risk (Fail/Withdrawn versus Pass/Distinction) and the original four-category result.

## Run the interactive console predictor

From this folder, install the console dependencies and run `main.py`. It starts the predictor by default and does not start a web server or open a browser:

```powershell
python -m pip install -r requirements-console.txt
python main.py
```

You can also run the predictor explicitly with `python main.py predict`. It asks for a hypothetical learner profile and prints the performance category and probabilities. Press Enter to use each default; enter `y` to score another profile.

## Run the full training pipeline

Obtain the official OULAD CSV archive from [UCI](https://archive.ics.uci.edu/dataset/349/open%2Buniversity%2B), place its CSV files in `data/original/`, install full dependencies, and run:

```powershell
python -m pip install -r requirements.txt
python main.py train
```

The pipeline writes metrics, predictions, and figures under `outputs/`; fitted pipelines are saved in `models/`. Student-level predictions and split assignments are generated locally and must not be published. See [data/README.md](data/README.md) for attribution and placement details.

## Method

- Source inspection validates schemas, row counts, data types, missing values, duplicates, enrollment keys, outcomes, and cutoff eligibility.
- Eligibility is recalculated separately for each cutoff. A known registration day must be on or before the cutoff; a withdrawal on/before the cutoff is ineligible; a withdrawn record with unknown unregistration timing is excluded as ambiguous.
- Predictors use module, presentation, registration timing, cutoff-limited clicks and active days, enrolled days, engagement rates, log clicks, prior attempts, studied credits, and assessment due/submission participation observed by cutoff. Assessment scores are excluded because score publication dates are unavailable. Final result and unregistration are never predictors. Student ID is only used as a group key.
- `StratifiedGroupKFold` creates approximate 70/15/15 partitions grouped by student and stratified by final result. Candidate models are evaluated with training-only grouped CV.
- OULAD represents one UK university. Results do not establish generalization to other institutions, causal effects, intervention effectiveness, or fair treatment of protected groups. Predictions are for research demonstration and require human review.

## Project contents

`main.py` contains both the interactive terminal predictor and training pipeline; `src/` contains training modules; `models/` contains fitted day-28 pipelines; `outputs/metrics/` and `outputs/figures/` contain aggregate evaluation artifacts. Raw data and the IEEE research paper are not part of this repository.
