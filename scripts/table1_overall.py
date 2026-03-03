#!/usr/bin/env python3
"""Table 1: Performance of all methods based on all subjects.

Computes SE, SP, PPV, LR+, LR- for each method at both
epoch and individual R-peak levels using raw per-R-peak adRRI.

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
from adarri.detector import THETA_EPOCH, THETA_RPEAK


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

    # Max raw adRRI per epoch (excluding leading zero)
    raw_adrri_max = np.array([
        np.max(ep["adrri_raw_ms"][1:]) if len(ep["adrri_raw_ms"]) > 1 else 0
        for ep in all_epochs
    ])

    # Method A: find optimal threshold in log space
    log_max = np.log(np.maximum(raw_adrri_max, 1e-10))
    thresholds = np.linspace(np.min(log_max), np.max(log_max), 500)

    best_acc, best_th_idx = 0, 0
    for t_idx, th in enumerate(thresholds):
        preds = (log_max > th).astype(int)
        acc = np.sum(preds == all_labels) / len(all_labels)
        if acc > best_acc:
            best_acc = acc
            best_th_idx = t_idx

    preds_a = (log_max > thresholds[best_th_idx]).astype(int)
    metrics_a = compute_metrics(all_labels, preds_a)
    optimal_theta_ms = np.exp(thresholds[best_th_idx])
    print(f"  Optimal theta (adRRI): {optimal_theta_ms:.0f} ms "
          f"(paper: {THETA_EPOCH} ms)")
    print(format_table_row(metrics_a, "Method A (ADARRI)"))

    # Method B (Berntson) - per-epoch threshold (matching MATLAB fcn_Sunils_data)
    preds_b = np.zeros(len(all_epochs), dtype=int)
    for j, ep in enumerate(all_epochs):
        adrri = ep["adrri_raw_ms"]
        iqr_val = scipy_iqr(adrri)
        med_val = np.median(adrri)
        MEDn = iqr_val / 2.0 * 3.32
        MADa = (med_val - 2.9 * iqr_val) / 3.0
        th_b_j = (abs(MEDn) + abs(MADa)) / 2.0
        if np.max(adrri) > th_b_j:
            preds_b[j] = 1
    metrics_b = compute_metrics(all_labels, preds_b)
    print(format_table_row(metrics_b, "Method B (Berntson)"))

    # Method C (Clifford) - use raw RRI
    preds_c = np.zeros(len(all_epochs), dtype=int)
    for j, ep in enumerate(all_epochs):
        rri_ms = ep["rri_raw_ms"]
        times_ms = ep["times_raw"] * 1000
        flags = detect_artifacts_clifford(times_ms, rri_ms, pc=80)
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

    # Get per-R-peak raw adRRI values and labels
    all_adrri_peaks = []
    all_rpeak_labels = []
    for j, ep in enumerate(all_epochs):
        adrri = ep["adrri_raw_ms"]  # Raw per-R-peak adRRI with leading 0
        all_adrri_peaks.append(adrri)
        all_rpeak_labels.append(np.full(len(adrri), all_labels[j], dtype=int))

    all_adrri_peaks = np.concatenate(all_adrri_peaks)
    all_rpeak_labels = np.concatenate(all_rpeak_labels)

    # Method A at R-peak level: sweep thresholds
    rpeak_thresholds = np.linspace(0, np.percentile(all_adrri_peaks, 99), 500)
    best_acc, best_th = 0, 0
    for th in rpeak_thresholds:
        preds = (all_adrri_peaks > th).astype(int)
        acc = np.sum(preds == all_rpeak_labels) / len(all_rpeak_labels)
        if acc > best_acc:
            best_acc = acc
            best_th = th

    preds_a_rp = (all_adrri_peaks > best_th).astype(int)
    metrics_a_rp = balanced_bootstrap_metrics(all_rpeak_labels, preds_a_rp)
    print(f"  Optimal theta (R-peak): {best_th:.0f} ms (paper: {THETA_RPEAK} ms)")
    print(format_table_row(metrics_a_rp, "Method A (ADARRI)"))

    # Method B at R-peak level - per-epoch threshold applied to R-peaks
    all_b_preds = []
    for j, ep in enumerate(all_epochs):
        adrri = ep["adrri_raw_ms"]
        iqr_val = scipy_iqr(adrri)
        med_val = np.median(adrri)
        MEDn = iqr_val / 2.0 * 3.32
        MADa = (med_val - 2.9 * iqr_val) / 3.0
        th_b_j = (abs(MEDn) + abs(MADa)) / 2.0
        all_b_preds.append((adrri > th_b_j).astype(int))
    preds_b_rp = np.concatenate(all_b_preds)
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
            "theta_A_ms": optimal_theta_ms,
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
