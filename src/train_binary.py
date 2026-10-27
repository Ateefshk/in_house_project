"""Train and compare binary risk models using group-only model selection."""
from training import fit_task, candidates
from models import EXP

def run_binary(features, assignments):
    return fit_task("binary",features,assignments)

if __name__ == "__main__":
    raise SystemExit("Run this module through main.py so the shared grouped split is used.")
