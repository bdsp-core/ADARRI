#!/usr/bin/env python3
"""Figure 2: Kernel probability density estimation of log(adRRI).

Two-panel figure showing:
  (top) Histogram/KDE of log(abs(diff(RRI))) for valid vs artifact epochs
  (bottom) Histogram/KDE of log(RRI) for valid vs artifact epochs

Usage:
    python scripts/figure2_kde_distributions.py --features data/processed/features.pkl --output outputs/
"""

import argparse
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def make_figure2(features_path, output_dir):
    """Generate Figure 2 from the paper."""
    os.makedirs(output_dir, exist_ok=True)

    with open(features_path, "rb") as f:
        data = pickle.load(f)

    X = data["X"]
    Y = data["Y"]

    # Pool all adRRI and RRI values by label
    all_adrri_valid = []
    all_adrri_artifact = []
    all_rri_valid = []
    all_rri_artifact = []

    for pt_idx in range(len(X)):
        epochs = X[pt_idx]
        labels = Y[pt_idx]
        for j, epoch in enumerate(epochs):
            adrri = epoch["adrri_ms"]
            rri = epoch["rri_ms"]
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

    # Log transform (avoid log(0))
    log_adrri_valid = np.log(np.maximum(all_adrri_valid, 1e-10))
    log_adrri_artifact = np.log(np.maximum(all_adrri_artifact, 1e-10))
    log_rri_valid = np.log(np.maximum(all_rri_valid, 1e-10))
    log_rri_artifact = np.log(np.maximum(all_rri_artifact, 1e-10))

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Top panel: log(adRRI) distributions
    ax = axes[0]
    all_log_adrri = np.concatenate([log_adrri_valid, log_adrri_artifact])
    bins = np.linspace(np.min(all_log_adrri), np.max(all_log_adrri), 200)

    h0, _ = np.histogram(log_adrri_valid, bins=bins)
    h1, _ = np.histogram(log_adrri_artifact, bins=bins)
    h0 = h0 / h0.sum()
    h1 = h1 / h1.sum()
    bin_centers = (bins[:-1] + bins[1:]) / 2

    ax.fill_between(bin_centers, h0, alpha=0.3, color="blue", label="Valid peaks")
    ax.plot(bin_centers, h0, "b-", linewidth=1)
    ax.fill_between(bin_centers, h1, alpha=0.3, color="red", label="Artifact peaks")
    ax.plot(bin_centers, h1, "r-", linewidth=1)
    ax.set_xlabel("log(adRRI) [log(ms)]")
    ax.set_ylabel("Probability Density")
    ax.set_title("Log of absolute difference in RR interval")
    ax.legend()

    # Bottom panel: log(RRI) distributions
    ax = axes[1]
    all_log_rri = np.concatenate([log_rri_valid, log_rri_artifact])
    bins = np.linspace(np.min(all_log_rri), np.max(all_log_rri), 200)

    h0, _ = np.histogram(log_rri_valid, bins=bins)
    h1, _ = np.histogram(log_rri_artifact, bins=bins)
    h0 = h0 / h0.sum()
    h1 = h1 / h1.sum()
    # Plot log of PDF as in MATLAB code
    h0_log = np.log(np.maximum(h0, 1e-10))
    h1_log = np.log(np.maximum(h1, 1e-10))
    bin_centers = (bins[:-1] + bins[1:]) / 2

    ax.fill_between(bin_centers, h0_log, alpha=0.3, color="blue", label="Valid peaks")
    ax.plot(bin_centers, h0_log, "b-", linewidth=1)
    ax.fill_between(bin_centers, h1_log, alpha=0.3, color="red", label="Artifact peaks")
    ax.plot(bin_centers, h1_log, "r-", linewidth=1)
    ax.set_xlabel("log(RRI) [log(ms)]")
    ax.set_ylabel("Log of Probability Density")
    ax.set_title("Log of RR interval")
    ax.legend()

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
