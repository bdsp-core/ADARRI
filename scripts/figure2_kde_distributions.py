#!/usr/bin/env python3
"""Figure 2: Kernel probability density estimation of log(adRRI).

Two-panel figure showing:
  (top) KDE of log(max adRRI per epoch) for valid vs artifact epochs
  (bottom) KDE of log(RRI) for valid vs artifact epochs

Matches the MATLAB fcnPlotKernelPDFestimatesAbsDif.m which plots
log(max(adRRI)) per epoch, and fcnPlotKernelPDFestimatesRRI.m which
plots log(RRI) for all individual values.

Usage:
    python scripts/figure2_kde_distributions.py --features data/processed/features.pkl --output outputs/
"""

import argparse
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def make_figure2(features_path, output_dir):
    """Generate Figure 2 from the paper."""
    os.makedirs(output_dir, exist_ok=True)

    with open(features_path, "rb") as f:
        data = pickle.load(f)

    X = data["X"]
    Y = data["Y"]

    # Collect max raw adRRI per epoch and all individual RRI values
    max_adrri_valid = []
    max_adrri_artifact = []
    all_rri_valid = []
    all_rri_artifact = []

    for pt_idx in range(len(X)):
        epochs = X[pt_idx]
        labels = Y[pt_idx]
        for j, epoch in enumerate(epochs):
            # Max raw adRRI per epoch (excluding leading zero)
            adrri = epoch["adrri_raw_ms"]
            max_val = np.max(adrri[1:]) if len(adrri) > 1 else 0
            rri = epoch["rri_raw_ms"]

            if labels[j] == 0:
                max_adrri_valid.append(max_val)
                all_rri_valid.append(rri)
            else:
                max_adrri_artifact.append(max_val)
                all_rri_artifact.append(rri)

    max_adrri_valid = np.array(max_adrri_valid)
    max_adrri_artifact = np.array(max_adrri_artifact)
    all_rri_valid = np.concatenate(all_rri_valid)
    all_rri_artifact = np.concatenate(all_rri_artifact)

    # Log-transform (filter out zero/negative values)
    log_adrri_valid = np.log(max_adrri_valid[max_adrri_valid > 0])
    log_adrri_artifact = np.log(max_adrri_artifact[max_adrri_artifact > 0])
    log_rri_valid = np.log(all_rri_valid[all_rri_valid > 0])
    log_rri_artifact = np.log(all_rri_artifact[all_rri_artifact > 0])

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # --- Top panel: log(max adRRI per epoch) ---
    ax = axes[0]
    all_log = np.concatenate([log_adrri_valid, log_adrri_artifact])
    xmin, xmax = np.min(all_log), np.max(all_log)
    xx = np.linspace(xmin, xmax, 1000)

    kde_valid = gaussian_kde(log_adrri_valid)
    kde_artifact = gaussian_kde(log_adrri_artifact)

    y_valid = kde_valid(xx)
    y_artifact = kde_artifact(xx)

    ax.fill_between(xx, y_valid, alpha=0.3, color="blue")
    ax.plot(xx, y_valid, "b-", linewidth=1.5, label="Valid epochs")
    ax.fill_between(xx, y_artifact, alpha=0.3, color="red")
    ax.plot(xx, y_artifact, "r-", linewidth=1.5, label="Artifact epochs")

    # Find and mark optimal threshold
    all_max = np.concatenate([max_adrri_valid, max_adrri_artifact])
    all_labels = np.concatenate([np.zeros(len(max_adrri_valid)),
                                 np.ones(len(max_adrri_artifact))])
    log_all = np.log(np.maximum(all_max, 1e-10))
    thresholds = np.linspace(np.min(log_all), np.max(log_all), 1000)
    best_acc, best_th = 0, thresholds[0]
    for th in thresholds:
        preds = (log_all > th).astype(int)
        acc = np.sum(preds == all_labels) / len(all_labels)
        if acc > best_acc:
            best_acc = acc
            best_th = th

    ymax = max(np.max(y_valid), np.max(y_artifact))
    ax.axvline(x=best_th, color="black", linestyle="--", linewidth=1.5,
               label=f"$\\theta$ = {np.exp(best_th):.0f} ms")
    ax.set_xlabel("log(max adRRI) [log(ms)]")
    ax.set_ylabel("Density")
    ax.set_title("Kernel density estimate: log(max |$\\Delta$RRI|) per epoch")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)

    print(f"adRRI valid: n={len(log_adrri_valid)}, "
          f"median max={np.median(max_adrri_valid):.1f}ms")
    print(f"adRRI artifact: n={len(log_adrri_artifact)}, "
          f"median max={np.median(max_adrri_artifact):.1f}ms")
    print(f"Optimal threshold: {np.exp(best_th):.0f}ms (paper: 276ms)")

    # --- Bottom panel: log(RRI) distributions ---
    ax = axes[1]
    all_log_rri = np.concatenate([log_rri_valid, log_rri_artifact])
    xmin, xmax = np.percentile(all_log_rri, 0.5), np.percentile(all_log_rri, 99.5)
    xx = np.linspace(xmin, xmax, 500)

    kde_valid_rri = gaussian_kde(log_rri_valid, bw_method=0.1)
    kde_artifact_rri = gaussian_kde(log_rri_artifact, bw_method=0.1)

    y_valid_rri = kde_valid_rri(xx)
    y_artifact_rri = kde_artifact_rri(xx)

    ax.fill_between(xx, y_valid_rri, alpha=0.3, color="blue")
    ax.plot(xx, y_valid_rri, "b-", linewidth=1.5, label="Valid epochs")
    ax.fill_between(xx, y_artifact_rri, alpha=0.3, color="red")
    ax.plot(xx, y_artifact_rri, "r-", linewidth=1.5, label="Artifact epochs")
    ax.set_xlabel("log(RRI) [log(ms)]")
    ax.set_ylabel("Density")
    ax.set_title("Kernel density estimate: log(RRI)")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    outpath = os.path.join(output_dir, "figure2_kde_distributions.png")
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    print(f"Saved Figure 2 to {outpath}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="data/processed/features.pkl")
    parser.add_argument("--output", default="outputs/")
    args = parser.parse_args()
    make_figure2(args.features, args.output)
