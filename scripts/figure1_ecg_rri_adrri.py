#!/usr/bin/env python3
"""Figure 1: Overview of RRI and adRRI derived from R-peak detection.

Three-panel figure showing:
  (a) ECG signal with detected R-peaks labeled
  (b) RRI time series showing valid (blue) vs artifact (red)
  (c) adRRI showing clear separation between valid and artifact peaks

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
from adarri.rri import compute_rri, compute_adrri


def make_figure1(data_dir, output_dir, patient_idx=0, epoch_idx=0):
    """Generate Figure 1 from the paper."""
    os.makedirs(output_dir, exist_ok=True)
    patients = load_all_patients(data_dir)

    # Use an artifact epoch for demonstration (shows both valid and artifact regions)
    patient = patients[patient_idx]
    ecg = patient["artifact_epochs"][epoch_idx]
    fs = 240
    time_ms = np.arange(len(ecg)) / fs * 1000

    # Detect R-peaks
    r_peaks = detect_r_peaks(ecg, fs)
    rri, rri_times = compute_rri(r_peaks, fs)
    rri_ms = rri * 1000
    rri_times_ms = rri_times * 1000
    adrri = compute_adrri(rri_ms)
    adrri_times_ms = rri_times_ms[:-1]

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=False)

    # (a) ECG signal with R-peak markers
    ax = axes[0]
    ax.plot(time_ms, ecg, "k", linewidth=0.5)
    ax.plot(time_ms[r_peaks], ecg[r_peaks], "rv", markersize=6, label="R-peaks")
    for i, rp in enumerate(r_peaks):
        ax.annotate(f"R$_{{{i+1}}}$", (time_ms[rp], ecg[rp]),
                     textcoords="offset points", xytext=(0, 10),
                     fontsize=7, ha="center")
    ax.set_ylabel("Amplitude")
    ax.set_title("(a) ECG signal with detected R-peaks")
    ax.legend(loc="upper right")

    # (b) RRI time series
    ax = axes[1]
    ax.plot(rri_times_ms, rri_ms, "b.-", markersize=4, linewidth=1)
    for i in range(len(rri_ms)):
        ax.annotate(f"B$_{{{i}}}$", (rri_times_ms[i], rri_ms[i]),
                     textcoords="offset points", xytext=(0, 8),
                     fontsize=7, ha="center", color="blue")
    ax.set_ylabel("RRI (ms)")
    ax.set_title("(b) RR interval time series")

    # (c) adRRI
    ax = axes[2]
    ax.plot(adrri_times_ms, adrri, "r.-", markersize=4, linewidth=1)
    for i in range(len(adrri)):
        ax.annotate(f"A$_{{{i}}}$", (adrri_times_ms[i], adrri[i]),
                     textcoords="offset points", xytext=(0, 8),
                     fontsize=7, ha="center", color="red")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("adRRI (ms)")
    ax.set_title("(c) Absolute difference of adjacent RR intervals")

    plt.tight_layout()
    outpath = os.path.join(output_dir, "figure1_ecg_rri_adrri.png")
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    print(f"Saved Figure 1 to {outpath}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/")
    parser.add_argument("--output", default="outputs/")
    parser.add_argument("--patient", type=int, default=0)
    parser.add_argument("--epoch", type=int, default=0)
    args = parser.parse_args()
    make_figure1(args.data_dir, args.output, args.patient, args.epoch)
