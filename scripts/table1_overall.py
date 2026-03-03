#!/usr/bin/env python3
"""Table 1: Performance of all methods based on all subjects.

Computes SE, SP, PPV, LR+, LR- for each method at both
epoch and individual R-peak levels.

Usage:
    python scripts/table1_overall.py --features data/processed/features.pkl --output outputs/
"""

import argparse
import os
import pickle
import sys

import numpy as np
from scipy.stats import iqr as scipy_iqr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from adarri.evaluation import compute_metrics, balanced_bootstrap_metrics, format_table_row
from adarri.clifford import detect_artifacts_clifford


def compute_table1(features_path, output_dir):
    """Generate Table 1."""
    os.makedirs(output_dir, exist_ok=True)

    with open(features_path, "rb") as f:
        data = pickle.load(f)

    # Pool all epochs
    all_epochs = []
    all_labels = []
    for pt_idx in range(len(data["X"])):
        all_epochs.extend(data["X"][pt_idx])
        all_labels.extend(data["Y"][pt_idx].tolist())

    all_labels = np.array(all_labels)

    # ===== EPOCH EVALUATION =====
    print("=" * 70)
    print("TABLE 1: Performance of all methods based on all subjects")
    print("=" * 70)
    print("\n--- Epoch Evaluation ---\n")

    # Find optimal threshold for Method A (adRRI)
    all_log_adrri_max = np.array([np.log(np.max(ep["adrri_ms"])) for ep in all_epochs])
    thresholds = np.linspace(np.min(all_log_adrri_max), np.max(all_log_adrri_max), 500)

    best_acc, best_th_idx = 0, 0
    for t_idx, th in enumerate(thresholds):
        preds = (all_log_adrri_max > th).astype(int)
        acc = np.sum(preds == all_labels) / len(all_labels)
        if acc > best_acc:
            best_acc = acc
            best_th_idx = t_idx

    # Method A predictions at optimal threshold
    preds_a = (all_log_adrri_max > thresholds[best_th_idx]).astype(int)
    metrics_a = compute_metrics(all_labels, preds_a)
    print(f"  Optimal theta (adRRI): {np.exp(thresholds[best_th_idx]):.0f} ms "
          f"(paper: 276 ms)")
    print(format_table_row(metrics_a, "Method A (ADARRI)"))

    # Method B (Berntson)
    all_adrri_pooled = np.concatenate([ep["adrri_ms"] for ep in all_epochs])
    iqr_val = scipy_iqr(all_adrri_pooled)
    med_val = np.median(all_adrri_pooled)
    MEDn = iqr_val / 2.0 * 3.32
    MADa = (med_val - 2.9 * iqr_val) / 3.0
    th_b = (abs(MEDn) + abs(MADa)) / 2.0

    preds_b = np.array([1 if np.max(ep["adrri_ms"]) > th_b else 0 for ep in all_epochs])
    metrics_b = compute_metrics(all_labels, preds_b)
    print(format_table_row(metrics_b, "Method B (Berntson)"))

    # Method C (Clifford)
    preds_c = np.zeros(len(all_epochs), dtype=int)
    for j, ep in enumerate(all_epochs):
        rri_ms = ep["rri_ms"]
        times = np.arange(len(rri_ms), dtype=np.float64)
        flags = detect_artifacts_clifford(times, rri_ms, pc=80)
        if np.any(flags):
            preds_c[j] = 1
    metrics_c = compute_metrics(all_labels, preds_c)
    print(format_table_row(metrics_c, "Method C (Clifford)"))

    # Combined AC
    preds_ac = ((preds_a == 1) | (preds_c == 1)).astype(int)
    metrics_ac = compute_metrics(all_labels, preds_ac)
    print(format_table_row(metrics_ac, "Method AC"))

    # Combined BC
    preds_bc = ((preds_b == 1) | (preds_c == 1)).astype(int)
    metrics_bc = compute_metrics(all_labels, preds_bc)
    print(format_table_row(metrics_bc, "Method BC"))

    # ===== R-PEAK EVALUATION =====
    print("\n--- Individual R-peak Evaluation ---\n")

    # Get per-R-peak data
    all_adrri_peaks = []
    all_rpeak_labels = []
    for j, ep in enumerate(all_epochs):
        adrri = ep["adrri_ms"]
        adrri_with_leading = np.concatenate([[0], adrri])
        all_adrri_peaks.append(adrri_with_leading)
        all_rpeak_labels.append(np.full(len(adrri_with_leading), all_labels[j], dtype=int))

    all_adrri_peaks = np.concatenate(all_adrri_peaks)
    all_rpeak_labels = np.concatenate(all_rpeak_labels)

    # Method A at R-peak level
    rpeak_thresholds = np.linspace(0, np.max(all_adrri_peaks) * 0.5, 500)
    best_acc, best_th = 0, 0
    for th in rpeak_thresholds:
        preds = (all_adrri_peaks > th).astype(int)
        acc = np.sum(preds == all_rpeak_labels) / len(all_rpeak_labels)
        if acc > best_acc:
            best_acc = acc
            best_th = th

    preds_a_rp = (all_adrri_peaks > best_th).astype(int)
    metrics_a_rp = balanced_bootstrap_metrics(all_rpeak_labels, preds_a_rp)
    print(f"  Optimal theta (R-peak): {best_th:.0f} ms (paper: 85 ms)")
    print(format_table_row(metrics_a_rp, "Method A (ADARRI)"))

    # Method B at R-peak level
    preds_b_rp = (all_adrri_peaks > th_b).astype(int)
    metrics_b_rp = balanced_bootstrap_metrics(all_rpeak_labels, preds_b_rp)
    print(format_table_row(metrics_b_rp, "Method B (Berntson)"))

    print(f"\nTotal R-peak detections: {len(all_rpeak_labels)}")
    print(f"  Valid: {np.sum(all_rpeak_labels == 0)}")
    print(f"  Artifact: {np.sum(all_rpeak_labels == 1)}")

    # Save results
    results = {
        "epoch": {
            "Method_A": metrics_a,
            "Method_B": metrics_b,
            "Method_C": metrics_c,
            "Method_AC": metrics_ac,
            "Method_BC": metrics_bc,
            "theta_A_ms": np.exp(thresholds[best_th_idx]),
        },
        "rpeak": {
            "Method_A": metrics_a_rp,
            "Method_B": metrics_b_rp,
            "theta_A_ms": best_th,
        },
    }

    outpath = os.path.join(output_dir, "table1_results.pkl")
    with open(outpath, "wb") as f:
        pickle.dump(results, f)
    print(f"\nSaved Table 1 results to {outpath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="data/processed/features.pkl")
    parser.add_argument("--output", default="outputs/")
    args = parser.parse_args()
    compute_table1(args.features, args.output)
