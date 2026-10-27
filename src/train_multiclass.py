"""Train and compare four-category performance models using grouped validation."""
from training import fit_task, candidates
from models import EXP

def run_multiclass(features, assignments):
    return fit_task("multiclass",features,assignments)

if __name__ == "__main__":
    raise SystemExit("Run this module through main.py so the shared grouped split is used.")
