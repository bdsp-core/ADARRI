#!/usr/bin/env python3
"""Figure 3: ROC and Precision-Recall curves for epoch-level evaluation.

Compares all methods (A, B, C, AC, BC) at the epoch level.

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


def _classify_all_epochs_method_a(epochs, thresholds):
    """Method A: classify epochs using adRRI threshold sweep."""
    n_epochs = len(epochs)
    n_th = len(thresholds)
    predictions = np.zeros((n_th, n_epochs), dtype=int)

    # Get max adRRI per epoch
    max_adrri = np.array([np.max(ep["adrri_ms"]) for ep in epochs])

    for t_idx, th in enumerate(thresholds):
        predictions[t_idx] = (max_adrri > np.exp(th)).astype(int)

    return predictions


def _classify_all_epochs_method_b(epochs, labels):
    """Method B (Berntson): single operating point."""
    # Pool all adRRI to compute threshold
    all_adrri = np.concatenate([ep["adrri_ms"] for ep in epochs])
    iqr_val = scipy_iqr(all_adrri)
    med_val = np.median(all_adrri)
    MEDn = iqr_val / 2.0 * 3.32
    MADa = (med_val - 2.9 * iqr_val) / 3.0
    threshold = (abs(MEDn) + abs(MADa)) / 2.0

    max_adrri = np.array([np.max(ep["adrri_ms"]) for ep in epochs])
    predictions = (max_adrri > threshold).astype(int)
    return predictions


def _classify_all_epochs_method_c(epochs, labels):
    """Method C (Clifford): single operating point based on 20% change."""
    from adarri.clifford import detect_artifacts_clifford

    predictions = np.zeros(len(epochs), dtype=int)
    for j, ep in enumerate(epochs):
        rri_ms = ep["rri_ms"]
        # Use indices as pseudo-times
        times = np.arange(len(rri_ms), dtype=np.float64)
        flags = detect_artifacts_clifford(times, rri_ms, pc=80)
        if np.any(flags):
            predictions[j] = 1
    return predictions


def make_figure3(features_path, output_dir):
    """Generate Figure 3."""
    os.makedirs(output_dir, exist_ok=True)

    with open(features_path, "rb") as f:
        data = pickle.load(f)

    # Pool all epochs across patients
    all_epochs = []
    all_labels = []
    for pt_idx in range(len(data["X"])):
        epochs = data["X"][pt_idx]
        labels = data["Y"][pt_idx]
        all_epochs.extend(epochs)
        all_labels.extend(labels.tolist())

    all_labels = np.array(all_labels)

    # Method A: sweep thresholds for ROC
    all_log_adrri_max = np.array([np.log(np.max(ep["adrri_ms"])) for ep in all_epochs])
    th_range = np.linspace(np.min(all_log_adrri_max), np.max(all_log_adrri_max), 500)

    se_a, sp_a, ppv_a = [], [], []
    for th in th_range:
        preds = (all_log_adrri_max > th).astype(int)
        tp = np.sum((all_labels == 1) & (preds == 1))
        fn = np.sum((all_labels == 1) & (preds == 0))
        tn = np.sum((all_labels == 0) & (preds == 0))
        fp = np.sum((all_labels == 0) & (preds == 1))
        se_a.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        sp_a.append(tn / (tn + fp) if (tn + fp) > 0 else 0)
        ppv_a.append(tp / (tp + fp) if (tp + fp) > 0 else 1)

    se_a, sp_a, ppv_a = np.array(se_a), np.array(sp_a), np.array(ppv_a)
    fpr_a = 1 - sp_a

    # Method B: single point
    preds_b = _classify_all_epochs_method_b(all_epochs, all_labels)
    tp_b = np.sum((all_labels == 1) & (preds_b == 1))
    fn_b = np.sum((all_labels == 1) & (preds_b == 0))
    tn_b = np.sum((all_labels == 0) & (preds_b == 0))
    fp_b = np.sum((all_labels == 0) & (preds_b == 1))
    se_b = tp_b / (tp_b + fn_b) if (tp_b + fn_b) > 0 else 0
    sp_b = tn_b / (tn_b + fp_b) if (tn_b + fp_b) > 0 else 0
    ppv_b = tp_b / (tp_b + fp_b) if (tp_b + fp_b) > 0 else 1
    fpr_b = 1 - sp_b

    # Method C: single point
    preds_c = _classify_all_epochs_method_c(all_epochs, all_labels)
    tp_c = np.sum((all_labels == 1) & (preds_c == 1))
    fn_c = np.sum((all_labels == 1) & (preds_c == 0))
    tn_c = np.sum((all_labels == 0) & (preds_c == 0))
    fp_c = np.sum((all_labels == 0) & (preds_c == 1))
    se_c = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0
    sp_c = tn_c / (tn_c + fp_c) if (tn_c + fp_c) > 0 else 0
    ppv_c = tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0 else 1
    fpr_c = 1 - sp_c

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ROC curve
    ax = axes[0]
    ax.plot(fpr_a, se_a, "b-", linewidth=2, label="Method A (ADARRI)")
    ax.plot(fpr_b, se_b, "gs", markersize=10, label="Method B (Berntson)")
    ax.plot(fpr_c, se_c, "r^", markersize=10, label="Method C (Clifford)")
    # Mark optimal point for Method A (max accuracy)
    acc_a = (se_a + sp_a) / 2
    best_idx = np.argmax(acc_a)
    ax.plot(fpr_a[best_idx], se_a[best_idx], "b*", markersize=15,
            label=f"Optimal A (SE={se_a[best_idx]:.2f}, SP={sp_a[best_idx]:.2f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("1 - Specificity (FPR)")
    ax.set_ylabel("Sensitivity (TPR)")
    ax.set_title("ROC Curve - Epoch Evaluation")
    ax.legend(loc="lower right")
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
    ax.legend(loc="lower left")
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
