"""
AutoResearch Experiment Runner
==============================
Runs systematic experiments by writing a config override file,
then importing it from train.py.

Usage: python3 run_experiments.py
"""

import subprocess
import time
import json
import os
from pathlib import Path

RESEARCH_DIR = Path(__file__).parent
RESULTS_TSV = RESEARCH_DIR / "results.tsv"
CONFIG_FILE = RESEARCH_DIR / "experiment_config.json"

# ── Experiment Definitions ────────────────────────────────────────────
EXPERIMENTS = [
    # Phase 1: GBR hyperparameter sweep
    {"desc": "GBR baseline n=200 d=4 lr=0.05",
     "MODEL_TYPE": "gbr", "GBR_N": 200, "GBR_DEPTH": 4, "GBR_LR": 0.05, "GBR_SUB": 0.8, "GBR_LEAF": 5},

    {"desc": "GBR n=500 d=4 lr=0.03",
     "MODEL_TYPE": "gbr", "GBR_N": 500, "GBR_DEPTH": 4, "GBR_LR": 0.03, "GBR_SUB": 0.8, "GBR_LEAF": 5},

    {"desc": "GBR n=300 d=3 lr=0.05",
     "MODEL_TYPE": "gbr", "GBR_N": 300, "GBR_DEPTH": 3, "GBR_LR": 0.05, "GBR_SUB": 0.8, "GBR_LEAF": 3},

    {"desc": "GBR n=500 d=5 lr=0.02 sub=0.7",
     "MODEL_TYPE": "gbr", "GBR_N": 500, "GBR_DEPTH": 5, "GBR_LR": 0.02, "GBR_SUB": 0.7, "GBR_LEAF": 5},

    {"desc": "GBR n=800 d=4 lr=0.01 sub=0.7",
     "MODEL_TYPE": "gbr", "GBR_N": 800, "GBR_DEPTH": 4, "GBR_LR": 0.01, "GBR_SUB": 0.7, "GBR_LEAF": 5},

    {"desc": "GBR n=400 d=6 lr=0.05 leaf=10",
     "MODEL_TYPE": "gbr", "GBR_N": 400, "GBR_DEPTH": 6, "GBR_LR": 0.05, "GBR_SUB": 0.8, "GBR_LEAF": 10},

    {"desc": "GBR n=600 d=5 lr=0.03 sub=0.75 leaf=4",
     "MODEL_TYPE": "gbr", "GBR_N": 600, "GBR_DEPTH": 5, "GBR_LR": 0.03, "GBR_SUB": 0.75, "GBR_LEAF": 4},

    {"desc": "GBR n=1000 d=3 lr=0.01 sub=0.8",
     "MODEL_TYPE": "gbr", "GBR_N": 1000, "GBR_DEPTH": 3, "GBR_LR": 0.01, "GBR_SUB": 0.8, "GBR_LEAF": 5},

    # Phase 2: Alternative models
    {"desc": "RF n=500 d=8 leaf=5",
     "MODEL_TYPE": "rf", "RF_N": 500, "RF_DEPTH": 8, "RF_LEAF": 5},

    {"desc": "RF n=500 d=12 leaf=3",
     "MODEL_TYPE": "rf", "RF_N": 500, "RF_DEPTH": 12, "RF_LEAF": 3},

    {"desc": "RF n=1000 d=10 leaf=3",
     "MODEL_TYPE": "rf", "RF_N": 1000, "RF_DEPTH": 10, "RF_LEAF": 3},

    {"desc": "Ridge alpha=1.0",
     "MODEL_TYPE": "ridge", "LINEAR_ALPHA": 1.0},

    {"desc": "Ridge alpha=0.1",
     "MODEL_TYPE": "ridge", "LINEAR_ALPHA": 0.1},

    {"desc": "Lasso alpha=0.01",
     "MODEL_TYPE": "lasso", "LINEAR_ALPHA": 0.01},

    {"desc": "ElasticNet alpha=0.1 l1=0.5",
     "MODEL_TYPE": "elastic", "LINEAR_ALPHA": 0.1, "ELASTIC_L1": 0.5},

    {"desc": "SVR C=10 rbf",
     "MODEL_TYPE": "svr", "SVR_C": 10.0, "SVR_EPS": 0.1},

    {"desc": "SVR C=50 rbf eps=0.05",
     "MODEL_TYPE": "svr", "SVR_C": 50.0, "SVR_EPS": 0.05},

    {"desc": "SVR C=100 rbf eps=0.01",
     "MODEL_TYPE": "svr", "SVR_C": 100.0, "SVR_EPS": 0.01},

    # Phase 3: Feature engineering
    {"desc": "Ridge poly=2 alpha=0.5",
     "MODEL_TYPE": "ridge", "LINEAR_ALPHA": 0.5, "POLY_DEGREE": 2},

    {"desc": "Lasso poly=2 alpha=0.01",
     "MODEL_TYPE": "lasso", "LINEAR_ALPHA": 0.01, "POLY_DEGREE": 2},
]


def run_experiment(exp, exp_id, best_mae):
    """Run a single experiment via config-driven train."""
    desc = exp["desc"]
    print(f"\n{'='*60}")
    print(f"Experiment #{exp_id}: {desc}")
    print(f"{'='*60}")

    # Write config file
    with open(CONFIG_FILE, "w") as f:
        json.dump(exp, f)

    # Run train_configurable.py
    start = time.time()
    result = subprocess.run(
        ["python3", str(RESEARCH_DIR / "train_configurable.py")],
        capture_output=True, text=True, timeout=300,
    )
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"  CRASH ({elapsed:.1f}s): {result.stderr[-200:]}")
        return None, "CRASH", 0

    # Parse output
    val_mae = None
    num_params = 0
    for line in result.stdout.split("\n"):
        if line.startswith("val_mae:"):
            val_mae = float(line.split(":")[1].strip())
        if line.startswith("num_params:"):
            num_params = int(line.split(":")[1].strip())

    if val_mae is None:
        print(f"  CRASH: no val_mae in output")
        return None, "CRASH", 0

    status = "KEPT" if val_mae < best_mae else "DISCARDED"
    print(f"  val_mae: {val_mae:.6f} (best: {best_mae:.6f}) -> {status} [{elapsed:.1f}s]")

    return val_mae, status, num_params


def main():
    # Initialize results.tsv
    if not RESULTS_TSV.exists():
        with open(RESULTS_TSV, "w") as f:
            f.write("commit\tval_mae\tnum_params\tstatus\tdescription\n")

    best_mae = float("inf")
    best_config = None
    total = len(EXPERIMENTS)

    for i, exp in enumerate(EXPERIMENTS):
        try:
            val_mae, status, num_params = run_experiment(exp, i + 1, best_mae)
        except subprocess.TimeoutExpired:
            print(f"  TIMEOUT")
            val_mae, status, num_params = None, "CRASH", 0

        if val_mae is not None:
            with open(RESULTS_TSV, "a") as f:
                f.write(f"exp_{i+1:03d}\t{val_mae:.6f}\t{num_params}\t{status}\t{exp['desc']}\n")

            if status == "KEPT":
                best_mae = val_mae
                best_config = exp

    # Summary
    print(f"\n{'='*60}")
    print(f"EXPERIMENT SUMMARY ({total} experiments)")
    print(f"{'='*60}")

    # Read results
    results = []
    with open(RESULTS_TSV) as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 4:
                results.append({"status": parts[3], "mae": float(parts[1]), "desc": parts[4] if len(parts) > 4 else ""})

    kept = [r for r in results if r["status"] == "KEPT"]
    disc = [r for r in results if r["status"] == "DISCARDED"]
    crash = [r for r in results if r["status"] == "CRASH"]

    print(f"Kept: {len(kept)}, Discarded: {len(disc)}, Crashed: {len(crash)}")
    print(f"Best val_mae: {best_mae:.6f}")
    if best_config:
        print(f"Best config: {best_config['desc']}")

    # Export for site
    print("\nExporting results...")
    subprocess.run(["python3", "export_results.py"], cwd=str(RESEARCH_DIR))

    # Run best config one final time for model.json export
    if best_config:
        print("Running best config for final model export...")
        with open(CONFIG_FILE, "w") as f:
            json.dump(best_config, f)
        subprocess.run(
            ["python3", str(RESEARCH_DIR / "train_configurable.py")],
            capture_output=True
        )

    # Cleanup
    if CONFIG_FILE.exists():
        CONFIG_FILE.unlink()

    print(f"\nDone! Best val_mae: {best_mae:.6f}")


if __name__ == "__main__":
    main()
