#!/usr/bin/env python3
"""Table 2: Performance of methods A, B, and C for individual subjects.

Computes per-subject metrics and reports medians and IQR.

Usage:
    python scripts/table2_per_subject.py --features data/processed/features.pkl --output outputs/
"""

import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd
from scipy.stats import iqr as scipy_iqr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from adarri.evaluation import compute_metrics
from adarri.clifford import detect_artifacts_clifford


def compute_table2(features_path, output_dir):
    """Generate Table 2."""
    os.makedirs(output_dir, exist_ok=True)

    with open(features_path, "rb") as f:
        data = pickle.load(f)

    X = data["X"]
    Y = data["Y"]
    patient_ids = data["patient_ids"]

    print("=" * 70)
    print("TABLE 2: Performance of methods A, B, and C for individual subjects")
    print("=" * 70)

    results_a = []
    results_b = []
    results_c = []

    for pt_idx in range(len(X)):
        epochs = X[pt_idx]
        labels = Y[pt_idx]

        if len(epochs) == 0 or np.sum(labels == 0) == 0 or np.sum(labels == 1) == 0:
            continue

        # Method A: find patient-specific optimal threshold
        all_log_adrri_max = np.array([np.log(np.max(ep["adrri_ms"])) for ep in epochs])
        thresholds = np.linspace(np.min(all_log_adrri_max), np.max(all_log_adrri_max), 500)
        best_acc, best_th_idx = 0, 0
        for t_idx, th in enumerate(thresholds):
            preds = (all_log_adrri_max > th).astype(int)
            acc = np.sum(preds == labels) / len(labels)
            if acc > best_acc:
                best_acc = acc
                best_th_idx = t_idx

        preds_a = (all_log_adrri_max > thresholds[best_th_idx]).astype(int)
        results_a.append(compute_metrics(labels, preds_a))

        # Method B: Berntson patient-specific threshold
        all_adrri = np.concatenate([ep["adrri_ms"] for ep in epochs])
        iqr_val = scipy_iqr(all_adrri)
        med_val = np.median(all_adrri)
        MEDn = iqr_val / 2.0 * 3.32
        MADa = (med_val - 2.9 * iqr_val) / 3.0
        th_b = (abs(MEDn) + abs(MADa)) / 2.0

        preds_b = np.array([1 if np.max(ep["adrri_ms"]) > th_b else 0 for ep in epochs])
        results_b.append(compute_metrics(labels, preds_b))

        # Method C: Clifford
        preds_c = np.zeros(len(epochs), dtype=int)
        for j, ep in enumerate(epochs):
            rri_ms = ep["rri_ms"]
            times = np.arange(len(rri_ms), dtype=np.float64)
            flags = detect_artifacts_clifford(times, rri_ms, pc=80)
            if np.any(flags):
                preds_c[j] = 1
        results_c.append(compute_metrics(labels, preds_c))

    # Summarize as median (IQR)
    def summarize(results, method_name):
        df = pd.DataFrame(results)
        print(f"\n{method_name}:")
        for col in ["SE", "SP", "PPV", "LR_plus", "LR_minus"]:
            med = df[col].median()
            q25 = df[col].quantile(0.25)
            q75 = df[col].quantile(0.75)
            if col in ["LR_plus", "LR_minus"]:
                print(f"  {col:>10s}: {med:.1f} ({q25:.1f} - {q75:.1f})")
            else:
                print(f"  {col:>10s}: {med:.0f}% ({q25:.0f}% - {q75:.0f}%)")
        return df

    print("\n--- Epoch Evaluation ---")
    df_a = summarize(results_a, "Method A (ADARRI)")
    df_b = summarize(results_b, "Method B (Berntson)")
    df_c = summarize(results_c, "Method C (Clifford)")

    # Save
    outpath = os.path.join(output_dir, "table2_results.pkl")
    with open(outpath, "wb") as f:
        pickle.dump({
            "Method_A": results_a,
            "Method_B": results_b,
            "Method_C": results_c,
        }, f)
    print(f"\nSaved Table 2 results to {outpath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="data/processed/features.pkl")
    parser.add_argument("--output", default="outputs/")
    args = parser.parse_args()
    compute_table2(args.features, args.output)
