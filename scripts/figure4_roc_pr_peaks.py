#!/usr/bin/env python3
"""Figure 4: ROC and Precision-Recall curves for individual R-peak evaluation.

Evaluates artifact detection at the individual R-peak level using raw
per-R-peak adRRI. Each R-peak inherits the label of its epoch.

Usage:
    python scripts/figure4_roc_pr_peaks.py --features data/processed/features.pkl --output outputs/
"""

import argparse
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import iqr as scipy_iqr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from adarri.detector import THETA_RPEAK


def make_figure4(features_path, output_dir):
    """Generate Figure 4."""
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

    # Get per-R-peak raw adRRI values and labels
    # Each R-peak inherits the label of its epoch
    all_adrri = []
    rpeak_labels = []

    for j, ep in enumerate(all_epochs):
        adrri = ep["adrri_raw_ms"]  # Raw per-R-peak adRRI with leading 0
        label = all_labels[j]
        all_adrri.append(adrri)
        rpeak_labels.append(np.full(len(adrri), label, dtype=int))

    all_adrri = np.concatenate(all_adrri)
    rpeak_labels = np.concatenate(rpeak_labels)

    print(f"Total R-peak detections: {len(all_adrri)}")
    print(f"  Valid: {np.sum(rpeak_labels == 0)} "
          f"({np.sum(rpeak_labels == 0)/len(rpeak_labels)*100:.1f}%)")
    print(f"  Artifact: {np.sum(rpeak_labels == 1)} "
          f"({np.sum(rpeak_labels == 1)/len(rpeak_labels)*100:.1f}%)")

    # --- Method A: sweep thresholds ---
    thresholds = np.linspace(0, np.percentile(all_adrri, 99), 500)

    se_a, sp_a, ppv_a = [], [], []
    for th in thresholds:
        preds = (all_adrri > th).astype(int)
        tp = np.sum((rpeak_labels == 1) & (preds == 1))
        fn = np.sum((rpeak_labels == 1) & (preds == 0))
        tn = np.sum((rpeak_labels == 0) & (preds == 0))
        fp = np.sum((rpeak_labels == 0) & (preds == 1))
        se_a.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        sp_a.append(tn / (tn + fp) if (tn + fp) > 0 else 0)
        ppv_a.append(tp / (tp + fp) if (tp + fp) > 0 else 1)

    se_a, sp_a, ppv_a = np.array(se_a), np.array(sp_a), np.array(ppv_a)
    fpr_a = 1 - sp_a

    # Optimal point (max balanced accuracy)
    acc_a = (se_a + sp_a) / 2
    best_idx = np.argmax(acc_a)
    optimal_theta = thresholds[best_idx]
    print(f"\nOptimal theta (R-peak level): {optimal_theta:.1f} ms "
          f"(paper: {THETA_RPEAK} ms)")
    print(f"  SE={se_a[best_idx]*100:.0f}%, SP={sp_a[best_idx]*100:.0f}%, "
          f"PPV={ppv_a[best_idx]*100:.0f}%")

    # --- Method B: Berntson per-epoch threshold ---
    all_b_preds = []
    all_b_labels = []
    for j, ep in enumerate(all_epochs):
        adrri = ep["adrri_raw_ms"]
        label = all_labels[j]
        iqr_val = scipy_iqr(adrri)
        med_val = np.median(adrri)
        MEDn = iqr_val / 2.0 * 3.32
        MADa = (med_val - 2.9 * iqr_val) / 3.0
        th_b_j = (abs(MEDn) + abs(MADa)) / 2.0
        all_b_preds.append((adrri > th_b_j).astype(int))
        all_b_labels.append(np.full(len(adrri), label, dtype=int))
    preds_b = np.concatenate(all_b_preds)
    rpeak_labels_b = np.concatenate(all_b_labels)
    tp_b = np.sum((rpeak_labels_b == 1) & (preds_b == 1))
    fn_b = np.sum((rpeak_labels_b == 1) & (preds_b == 0))
    tn_b = np.sum((rpeak_labels_b == 0) & (preds_b == 0))
    fp_b = np.sum((rpeak_labels_b == 0) & (preds_b == 1))
    se_b = tp_b / (tp_b + fn_b) if (tp_b + fn_b) > 0 else 0
    sp_b = tn_b / (tn_b + fp_b) if (tn_b + fp_b) > 0 else 0
    ppv_b = tp_b / (tp_b + fp_b) if (tp_b + fp_b) > 0 else 1
    print(f"Method B (per-epoch Berntson): "
          f"SE={se_b*100:.0f}%, SP={sp_b*100:.0f}%, PPV={ppv_b*100:.0f}%")

    # --- AUC ---
    sorted_idx = np.argsort(fpr_a)
    auc_a = np.trapezoid(se_a[sorted_idx], fpr_a[sorted_idx])

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ROC curve
    ax = axes[0]
    ax.plot(fpr_a, se_a, "b-", linewidth=2, label=f"Method A (AUC={auc_a:.3f})")
    ax.plot(1 - sp_b, se_b, "gs", markersize=10, label="Method B (Berntson)")
    ax.plot(fpr_a[best_idx], se_a[best_idx], "b*", markersize=15,
            label=f"Optimal A ($\\theta$={optimal_theta:.0f}ms)")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("1 - Specificity (FPR)")
    ax.set_ylabel("Sensitivity (TPR)")
    ax.set_title("ROC Curve - Individual R-peak Evaluation")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.grid(True, alpha=0.3)

    # Precision-Recall curve
    ax = axes[1]
    ax.plot(se_a, ppv_a, "b-", linewidth=2, label="Method A (ADARRI)")
    ax.plot(se_b, ppv_b, "gs", markersize=10, label="Method B (Berntson)")
    ax.plot(se_a[best_idx], ppv_a[best_idx], "b*", markersize=15, label="Optimal A")
    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision (PPV)")
    ax.set_title("Precision-Recall Curve - Individual R-peak Evaluation")
    ax.legend(loc="lower left", fontsize=9)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    outpath = os.path.join(output_dir, "figure4_roc_pr_peaks.png")
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    print(f"Saved Figure 4 to {outpath}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="data/processed/features.pkl")
    parser.add_argument("--output", default="outputs/")
    args = parser.parse_args()
    make_figure4(args.features, args.output)
