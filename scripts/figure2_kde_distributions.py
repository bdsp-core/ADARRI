#!/usr/bin/env python3
"""Figure 2: Kernel probability density estimation of log(adRRI).

Two-panel figure showing:
  (top) KDE of log(adRRI) for valid vs artifact R-peaks (raw per-R-peak)
  (bottom) KDE of log(RRI) for valid vs artifact R-peaks (raw per-R-peak)

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

    # Pool all raw per-R-peak adRRI and RRI values by label
    all_adrri_valid = []
    all_adrri_artifact = []
    all_rri_valid = []
    all_rri_artifact = []

    for pt_idx in range(len(X)):
        epochs = X[pt_idx]
        labels = Y[pt_idx]
        for j, epoch in enumerate(epochs):
            adrri = epoch["adrri_raw_ms"]
            rri = epoch["rri_raw_ms"]
            if labels[j] == 0:
                all_adrri_valid.append(adrri)
                all_rri_valid.append(rri)
            else:
                all_adrri_artifact.append(adrri)
                all_rri_artifact.append(rri)

    all_adrri_valid = np.concatenate(all_adrri_valid)
    all_adrri_artifact = np.concatenate(all_adrri_artifact)
    all_rri_valid = np.concatenate(all_rri_valid)
    all_rri_artifact = np.concatenate(all_rri_artifact)

    # Filter out zero/negative values for log transform
    adrri_valid_pos = all_adrri_valid[all_adrri_valid > 0]
    adrri_artifact_pos = all_adrri_artifact[all_adrri_artifact > 0]
    rri_valid_pos = all_rri_valid[all_rri_valid > 0]
    rri_artifact_pos = all_rri_artifact[all_rri_artifact > 0]

    log_adrri_valid = np.log(adrri_valid_pos)
    log_adrri_artifact = np.log(adrri_artifact_pos)
    log_rri_valid = np.log(rri_valid_pos)
    log_rri_artifact = np.log(rri_artifact_pos)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # --- Top panel: log(adRRI) distributions ---
    ax = axes[0]
    all_log_adrri = np.concatenate([log_adrri_valid, log_adrri_artifact])
    xmin, xmax = np.percentile(all_log_adrri, 0.5), np.percentile(all_log_adrri, 99.5)
    xx = np.linspace(xmin, xmax, 500)

    kde_valid = gaussian_kde(log_adrri_valid, bw_method=0.1)
    kde_artifact = gaussian_kde(log_adrri_artifact, bw_method=0.1)

    y_valid = kde_valid(xx)
    y_artifact = kde_artifact(xx)

    ax.fill_between(xx, y_valid, alpha=0.3, color="blue")
    ax.plot(xx, y_valid, "b-", linewidth=1.5, label="Valid R-peaks")
    ax.fill_between(xx, y_artifact, alpha=0.3, color="red")
    ax.plot(xx, y_artifact, "r-", linewidth=1.5, label="Artifact R-peaks")
    ax.set_xlabel("log(adRRI) [log(ms)]")
    ax.set_ylabel("Density")
    ax.set_title("Kernel density estimate: log of absolute difference in RR interval")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)

    print(f"adRRI valid: n={len(log_adrri_valid)}, "
          f"median={np.median(adrri_valid_pos):.1f}ms")
    print(f"adRRI artifact: n={len(log_adrri_artifact)}, "
          f"median={np.median(adrri_artifact_pos):.1f}ms")

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
    ax.plot(xx, y_valid_rri, "b-", linewidth=1.5, label="Valid R-peaks")
    ax.fill_between(xx, y_artifact_rri, alpha=0.3, color="red")
    ax.plot(xx, y_artifact_rri, "r-", linewidth=1.5, label="Artifact R-peaks")
    ax.set_xlabel("log(RRI) [log(ms)]")
    ax.set_ylabel("Density")
    ax.set_title("Kernel density estimate: log of RR interval")
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
