#!/usr/bin/env python3
"""Figure 3: ROC and Precision-Recall curves for epoch-level evaluation.

Compares all methods (A, B, C, AC, BC) at the epoch level using raw
per-R-peak adRRI. An epoch is classified as artifact if max(raw_adRRI) >
threshold.

Usage:
    python scripts/figure3_roc_pr_epochs.py --features data/processed/features.pkl --output outputs/
"""

import argparse
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import iqr as scipy_iqr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from adarri.clifford import detect_artifacts_clifford
from adarri.detector import THETA_EPOCH


def make_figure3(features_path, output_dir):
    """Generate Figure 3."""
    os.makedirs(output_dir, exist_ok=True)

    with open(features_path, "rb") as f:
        data = pickle.load(f)

    # Pool all epochs across patients
    all_epochs = []
    all_labels = []
    for pt_idx in range(len(data["X"])):
        all_epochs.extend(data["X"][pt_idx])
        all_labels.extend(data["Y"][pt_idx].tolist())

    all_labels = np.array(all_labels)

    # --- Method A: sweep thresholds on raw adRRI ---
    # Max raw adRRI per epoch (excluding leading zero)
    raw_adrri_max = np.array([
        np.max(ep["adrri_raw_ms"][1:]) if len(ep["adrri_raw_ms"]) > 1 else 0
        for ep in all_epochs
    ])

    # Sweep in log space (matching MATLAB: max(data) > exp(theta))
    log_max = np.log(np.maximum(raw_adrri_max, 1e-10))
    th_range = np.linspace(np.min(log_max), np.max(log_max), 500)

    se_a, sp_a, ppv_a = [], [], []
    for th in th_range:
        preds = (log_max > th).astype(int)
        tp = np.sum((all_labels == 1) & (preds == 1))
        fn = np.sum((all_labels == 1) & (preds == 0))
        tn = np.sum((all_labels == 0) & (preds == 0))
        fp = np.sum((all_labels == 0) & (preds == 1))
        se_a.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        sp_a.append(tn / (tn + fp) if (tn + fp) > 0 else 0)
        ppv_a.append(tp / (tp + fp) if (tp + fp) > 0 else 1)

    se_a, sp_a, ppv_a = np.array(se_a), np.array(sp_a), np.array(ppv_a)
    fpr_a = 1 - sp_a

    # Optimal point (max accuracy)
    acc_a = (se_a + sp_a) / 2
    best_idx = np.argmax(acc_a)
    optimal_theta_ms = np.exp(th_range[best_idx])
    print(f"Method A optimal threshold: {optimal_theta_ms:.0f} ms "
          f"(paper: {THETA_EPOCH} ms)")
    print(f"  SE={se_a[best_idx]*100:.0f}%, SP={sp_a[best_idx]*100:.0f}%, "
          f"PPV={ppv_a[best_idx]*100:.0f}%")

    # --- Method B (Berntson): per-epoch threshold (matching MATLAB fcn_Sunils_data) ---
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

    tp_b = np.sum((all_labels == 1) & (preds_b == 1))
    fn_b = np.sum((all_labels == 1) & (preds_b == 0))
    tn_b = np.sum((all_labels == 0) & (preds_b == 0))
    fp_b = np.sum((all_labels == 0) & (preds_b == 1))
    se_b = tp_b / (tp_b + fn_b) if (tp_b + fn_b) > 0 else 0
    sp_b = tn_b / (tn_b + fp_b) if (tn_b + fp_b) > 0 else 0
    ppv_b = tp_b / (tp_b + fp_b) if (tp_b + fp_b) > 0 else 1
    fpr_b = 1 - sp_b
    print(f"Method B (per-epoch Berntson): "
          f"SE={se_b*100:.0f}%, SP={sp_b*100:.0f}%, PPV={ppv_b*100:.0f}%")

    # --- Method C (Clifford): single operating point ---
    preds_c = np.zeros(len(all_epochs), dtype=int)
    for j, ep in enumerate(all_epochs):
        rri_ms = ep["rri_raw_ms"]
        times_ms = ep["times_raw"] * 1000
        flags = detect_artifacts_clifford(times_ms, rri_ms, pc=80)
        if np.any(flags):
            preds_c[j] = 1

    tp_c = np.sum((all_labels == 1) & (preds_c == 1))
    fn_c = np.sum((all_labels == 1) & (preds_c == 0))
    tn_c = np.sum((all_labels == 0) & (preds_c == 0))
    fp_c = np.sum((all_labels == 0) & (preds_c == 1))
    se_c = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0
    sp_c = tn_c / (tn_c + fp_c) if (tn_c + fp_c) > 0 else 0
    ppv_c = tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0 else 1
    fpr_c = 1 - sp_c
    print(f"Method C: SE={se_c*100:.0f}%, SP={sp_c*100:.0f}%, PPV={ppv_c*100:.0f}%")

    # --- Compute AUC ---
    sorted_idx = np.argsort(fpr_a)
    auc_a = np.trapezoid(se_a[sorted_idx], fpr_a[sorted_idx])

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ROC curve
    ax = axes[0]
    ax.plot(fpr_a, se_a, "b-", linewidth=2, label=f"Method A (AUC={auc_a:.3f})")
    ax.plot(fpr_b, se_b, "gs", markersize=10, label="Method B (Berntson)")
    ax.plot(fpr_c, se_c, "r^", markersize=10, label="Method C (Clifford)")
    ax.plot(fpr_a[best_idx], se_a[best_idx], "b*", markersize=15,
            label=f"Optimal A ($\\theta$={optimal_theta_ms:.0f}ms)")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("1 - Specificity (FPR)")
    ax.set_ylabel("Sensitivity (TPR)")
    ax.set_title("ROC Curve - Epoch Evaluation")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.grid(True, alpha=0.3)

    # Precision-Recall curve
    ax = axes[1]
    ax.plot(se_a, ppv_a, "b-", linewidth=2, label="Method A (ADARRI)")
    ax.plot(se_b, ppv_b, "gs", markersize=10, label="Method B (Berntson)")
    ax.plot(se_c, ppv_c, "r^", markersize=10, label="Method C (Clifford)")
    ax.plot(se_a[best_idx], ppv_a[best_idx], "b*", markersize=15, label="Optimal A")
    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision (PPV)")
    ax.set_title("Precision-Recall Curve - Epoch Evaluation")
    ax.legend(loc="lower left", fontsize=9)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    outpath = os.path.join(output_dir, "figure3_roc_pr_epochs.png")
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    print(f"Saved Figure 3 to {outpath}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="data/processed/features.pkl")
    parser.add_argument("--output", default="outputs/")
    args = parser.parse_args()
    make_figure3(args.features, args.output)
