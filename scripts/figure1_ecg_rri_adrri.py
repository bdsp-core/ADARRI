#!/usr/bin/env python3
"""Figure 1: Illustration of the ADARRI method.

Three-panel figure matching the paper's Figure 1:
  (a) ECG signal with R-peaks labeled (valid=blue diamonds, artifact=red X)
  (b) RRI: blue=valid-only RRI, red=all-peaks RRI (showing artifact effect)
  (c) adRRI: blue=valid-only adRRI, red=all-peaks adRRI

The key insight: one spurious R-peak (Rx) causes dramatic changes in
the RRI and adRRI sequences (red), while the valid-only sequences (blue)
remain stable.

Usage:
    python scripts/figure1_ecg_rri_adrri.py --data-dir data/ --output outputs/
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from adarri.io import load_all_patients
from adarri.peak_detection import detect_r_peaks
from adarri.rri import compute_rri


# Pre-selected epoch that closely matches the paper's Figure 1:
# Clean ECG, one small artifact peak (93 mV vs 465 mV valid median),
# very consistent valid RRI (~917 ms, CV=0.5%), dramatic adRRI (579 ms).
DEFAULT_PATIENT = 11
DEFAULT_EPOCH_TYPE = "artifact_epochs"
DEFAULT_EPOCH_IDX = 13
DEFAULT_START_SAMPLE = 7680
DEFAULT_ART_IDX = 3  # index of the artifact peak within the window


def make_figure1(data_dir, output_dir, patient_idx=None, epoch_type=None,
                 epoch_idx=None, start_sample=None, art_idx=None):
    """Generate Figure 1 from the paper."""
    os.makedirs(output_dir, exist_ok=True)
    patients = load_all_patients(data_dir)
    fs = 240
    window_samples = int(8000 / 1000.0 * fs)  # 8 seconds

    pt = patient_idx if patient_idx is not None else DEFAULT_PATIENT
    etype = epoch_type if epoch_type is not None else DEFAULT_EPOCH_TYPE
    ep = epoch_idx if epoch_idx is not None else DEFAULT_EPOCH_IDX
    start = start_sample if start_sample is not None else DEFAULT_START_SAMPLE
    art_k = art_idx if art_idx is not None else DEFAULT_ART_IDX

    ecg_full = patients[pt][etype][ep]
    r_peaks_full = detect_r_peaks(ecg_full, fs)

    # Extract window
    end = start + window_samples
    mask = (r_peaks_full >= start) & (r_peaks_full < end)
    r_peaks_abs = r_peaks_full[mask]
    r_peaks = r_peaks_abs - start  # relative to window
    ecg = ecg_full[start:end]
    n_peaks = len(r_peaks)

    if n_peaks < 3 or art_k >= n_peaks:
        print(f"ERROR: Invalid parameters (n_peaks={n_peaks}, art_idx={art_k})")
        return

    # Separate valid and artifact peaks
    all_peaks = r_peaks
    valid_peaks = np.delete(r_peaks, art_k)
    n_valid = len(valid_peaks)

    # Time axes in ms
    ecg_time_ms = np.arange(len(ecg)) / fs * 1000.0
    all_times_ms = all_peaks / fs * 1000.0
    valid_times_ms = valid_peaks / fs * 1000.0

    # RRI using ALL peaks (red "artifactual")
    all_rri_ms = np.diff(all_peaks) / fs * 1000.0
    all_rri_times_ms = all_times_ms[:-1]
    n_all_rri = len(all_rri_ms)

    # RRI using only VALID peaks (blue "valid")
    valid_rri_ms = np.diff(valid_peaks) / fs * 1000.0
    valid_rri_times_ms = valid_times_ms[:-1]
    n_valid_rri = len(valid_rri_ms)

    # adRRI for both
    all_adrri_ms = np.abs(np.diff(all_rri_ms))
    all_adrri_times_ms = all_rri_times_ms[:-1]

    valid_adrri_ms = np.abs(np.diff(valid_rri_ms))
    valid_adrri_times_ms = valid_rri_times_ms[:-1]

    # --- Create the figure ---
    fig, axes = plt.subplots(3, 1, figsize=(10, 8.5))
    plt.subplots_adjust(hspace=0.40, left=0.10, right=0.96, top=0.97, bottom=0.06)

    # ========== Panel (a): ECG ==========
    ax = axes[0]
    ax.plot(ecg_time_ms, ecg, "k-", linewidth=0.5)

    # Valid R-peaks: blue diamonds
    ax.plot(ecg_time_ms[valid_peaks], ecg[valid_peaks], "bD",
            markersize=7, markerfacecolor="blue", markeredgecolor="blue",
            zorder=5)

    # Artifact R-peak: red X
    art_peak = all_peaks[art_k]
    ax.plot(ecg_time_ms[art_peak], ecg[art_peak], "rx",
            markersize=10, markeredgewidth=2.5, zorder=5)

    # Label peaks: R_1, R_2, ... for valid; R_x for artifact
    valid_num = 0
    for i in range(n_peaks):
        t_ms = ecg_time_ms[all_peaks[i]]
        y_val = ecg[all_peaks[i]]
        if i == art_k:
            label = r"$\mathrm{R}_\mathrm{x}$"
            color = "red"
        else:
            valid_num += 1
            label = rf"$\mathrm{{R}}_{{{valid_num}}}$"
            color = "blue"
        ax.annotate(label, (t_ms, y_val),
                    textcoords="offset points", xytext=(0, 14),
                    fontsize=9, ha="center", color=color)

    ax.set_ylabel("Amplitude (mV)", fontsize=10)
    ax.text(0.01, 0.95, r"$\mathbf{a}$  ECG", transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="top")
    ax.set_xlim([0, ecg_time_ms[-1]])

    # ========== Panel (b): RRI ==========
    ax = axes[1]

    # Blue: valid-only RRI (dash-dot + diamonds)
    ax.plot(valid_rri_times_ms, valid_rri_ms, "b-.",
            linewidth=1.5, zorder=3, label='"Valid"')
    ax.plot(valid_rri_times_ms, valid_rri_ms, "bD",
            markersize=7, markerfacecolor="blue", markeredgecolor="blue",
            zorder=4)

    # Red: all-peaks RRI (dotted + X markers)
    ax.plot(all_rri_times_ms, all_rri_ms, "r:",
            linewidth=1.5, zorder=3, label='"Artifactual"')
    ax.plot(all_rri_times_ms, all_rri_ms, "rx",
            markersize=8, markeredgewidth=2, zorder=4)

    # B subscripts for valid (below points)
    for i in range(n_valid_rri):
        ax.annotate(rf"$\mathrm{{B}}_{{{i}}}$",
                    (valid_rri_times_ms[i], valid_rri_ms[i]),
                    textcoords="offset points", xytext=(8, -14),
                    fontsize=7, ha="center", color="blue")

    # B subscripts for all (above points)
    for i in range(n_all_rri):
        ax.annotate(rf"$\mathrm{{B}}_{{{i}}}$",
                    (all_rri_times_ms[i], all_rri_ms[i]),
                    textcoords="offset points", xytext=(8, 10),
                    fontsize=7, ha="center", color="red")

    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.set_ylabel("Time (ms)", fontsize=10)
    ax.text(0.01, 0.95, r"$\mathbf{b}$  RRI", transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="top")
    ax.set_xlim([0, ecg_time_ms[-1]])

    # ========== Panel (c): adRRI ==========
    ax = axes[2]

    # Blue: valid-only adRRI
    ax.plot(valid_adrri_times_ms, valid_adrri_ms, "b-.",
            linewidth=1.5, zorder=3)
    ax.plot(valid_adrri_times_ms, valid_adrri_ms, "bD",
            markersize=7, markerfacecolor="blue", markeredgecolor="blue",
            zorder=4)

    # Red: all-peaks adRRI
    ax.plot(all_adrri_times_ms, all_adrri_ms, "r:",
            linewidth=1.5, zorder=3)
    ax.plot(all_adrri_times_ms, all_adrri_ms, "rx",
            markersize=8, markeredgewidth=2, zorder=4)

    # A subscripts for valid (below)
    for i in range(len(valid_adrri_ms)):
        ax.annotate(rf"$\mathrm{{A}}_{{{i}}}$",
                    (valid_adrri_times_ms[i], valid_adrri_ms[i]),
                    textcoords="offset points", xytext=(8, -14),
                    fontsize=7, ha="center", color="blue")

    # A subscripts for all (above)
    for i in range(len(all_adrri_ms)):
        ax.annotate(rf"$\mathrm{{A}}_{{{i}}}$",
                    (all_adrri_times_ms[i], all_adrri_ms[i]),
                    textcoords="offset points", xytext=(8, 10),
                    fontsize=7, ha="center", color="red")

    ax.set_xlabel("Time (ms)", fontsize=10)
    ax.set_ylabel("Time (ms)", fontsize=10)
    ax.text(0.01, 0.95, r"$\mathbf{c}$  adRRI", transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="top")
    ax.set_xlim([0, ecg_time_ms[-1]])

    outpath = os.path.join(output_dir, "figure1_ecg_rri_adrri.png")
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    print(f"Saved Figure 1 to {outpath}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/")
    parser.add_argument("--output", default="outputs/")
    parser.add_argument("--patient", type=int, default=None)
    parser.add_argument("--epoch-type", default=None)
    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--art-idx", type=int, default=None)
    args = parser.parse_args()
    make_figure1(args.data_dir, args.output, args.patient, args.epoch_type,
                 args.epoch, args.start, args.art_idx)
