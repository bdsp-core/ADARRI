#!/usr/bin/env python3
"""Figure 1: Overview of RRI and adRRI derived from R-peak detection.

Three-panel figure matching the paper's style:
  (a) ECG signal with R-peaks labeled (valid=blue diamonds, artifact=red X's)
  (b) RRI time series with valid/artifact R-peaks distinguished
  (c) adRRI time series showing threshold separation

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
from adarri.detector import THETA_RPEAK


def find_good_artifact_epoch(patients):
    """Find an artifact epoch with a mix of valid and artifact-like R-peaks."""
    best_epoch = None
    best_score = -1

    for pt_idx, patient in enumerate(patients):
        for ep_idx, ecg in enumerate(patient["artifact_epochs"]):
            r_peaks = detect_r_peaks(ecg, 240)
            if len(r_peaks) < 6 or len(r_peaks) > 20:
                continue

            rri, times = compute_rri(r_peaks, 240)
            rri_ms = rri * 1000.0
            adrri_ms = np.concatenate([[0], np.abs(np.diff(rri_ms))])

            n_above = np.sum(adrri_ms > THETA_RPEAK)
            n_below = np.sum(adrri_ms <= THETA_RPEAK)

            # Want a good mix of flagged and unflagged peaks
            if n_above >= 2 and n_below >= 3:
                score = min(n_above, n_below)
                if score > best_score:
                    best_score = score
                    best_epoch = {
                        "ecg": ecg,
                        "r_peaks": r_peaks,
                        "rri_ms": rri_ms,
                        "times": times,
                        "adrri_ms": adrri_ms,
                        "patient_idx": pt_idx,
                        "epoch_idx": ep_idx,
                    }

    return best_epoch


def make_figure1(data_dir, output_dir, patient_idx=None, epoch_idx=None):
    """Generate Figure 1 from the paper."""
    os.makedirs(output_dir, exist_ok=True)
    patients = load_all_patients(data_dir)

    if patient_idx is not None and epoch_idx is not None:
        ecg = patients[patient_idx]["artifact_epochs"][epoch_idx]
        r_peaks = detect_r_peaks(ecg, 240)
        rri, times = compute_rri(r_peaks, 240)
        rri_ms = rri * 1000.0
        adrri_ms = np.concatenate([[0], np.abs(np.diff(rri_ms))])
        ep = {
            "ecg": ecg, "r_peaks": r_peaks, "rri_ms": rri_ms,
            "times": times, "adrri_ms": adrri_ms,
        }
    else:
        ep = find_good_artifact_epoch(patients)
        if ep is None:
            print("Could not find a suitable artifact epoch, using first artifact epoch")
            ecg = patients[0]["artifact_epochs"][0]
            r_peaks = detect_r_peaks(ecg, 240)
            rri, times = compute_rri(r_peaks, 240)
            rri_ms = rri * 1000.0
            adrri_ms = np.concatenate([[0], np.abs(np.diff(rri_ms))])
            ep = {
                "ecg": ecg, "r_peaks": r_peaks, "rri_ms": rri_ms,
                "times": times, "adrri_ms": adrri_ms,
            }

    ecg = ep["ecg"]
    r_peaks = ep["r_peaks"]
    rri_ms = ep["rri_ms"]
    times_s = ep["times"]
    adrri_ms = ep["adrri_ms"]
    fs = 240

    # Classify each R-peak: artifact if adRRI > theta
    # adrri_ms has N-1 entries (for N R-peaks); pad with False for last R-peak
    flags_rri = adrri_ms > THETA_RPEAK  # length N-1 (same as rri_ms)
    flags = np.concatenate([flags_rri, [False]])  # length N (same as r_peaks)

    # Time axes
    ecg_time_s = np.arange(len(ecg)) / fs
    rri_time_s = times_s
    adrri_time_s = times_s

    fig, axes = plt.subplots(3, 1, figsize=(12, 9))

    # --- (a) ECG signal with R-peak markers ---
    ax = axes[0]
    ax.plot(ecg_time_s, ecg, "k-", linewidth=0.5, alpha=0.7)

    valid_mask = ~flags[:len(r_peaks)]
    artifact_mask = flags[:len(r_peaks)]

    valid_peaks = r_peaks[valid_mask]
    artifact_peaks = r_peaks[artifact_mask]

    ax.plot(ecg_time_s[valid_peaks], ecg[valid_peaks], "bD",
            markersize=6, label="Valid R-peaks", markerfacecolor="blue")
    ax.plot(ecg_time_s[artifact_peaks], ecg[artifact_peaks], "rx",
            markersize=8, markeredgewidth=2, label="Artifact R-peaks")

    for i, rp in enumerate(r_peaks):
        ax.annotate(f"R$_{{{i+1}}}$", (ecg_time_s[rp], ecg[rp]),
                     textcoords="offset points", xytext=(0, 12),
                     fontsize=7, ha="center",
                     color="red" if flags[i] else "blue")

    ax.set_ylabel("Amplitude")
    ax.set_title("(a) ECG signal with detected R-peaks")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim([ecg_time_s[0], ecg_time_s[-1]])

    # --- (b) RRI time series ---
    ax = axes[1]

    for i in range(len(rri_ms)):
        color = "red" if flags_rri[i] else "blue"
        marker = "x" if flags_rri[i] else "D"
        ms = 8 if flags_rri[i] else 5
        ax.plot(rri_time_s[i], rri_ms[i], marker=marker, color=color,
                markersize=ms, markeredgewidth=1.5 if flags_rri[i] else 1)

    for i in range(len(rri_ms) - 1):
        if flags_rri[i] or flags_rri[i + 1]:
            ax.plot([rri_time_s[i], rri_time_s[i + 1]],
                    [rri_ms[i], rri_ms[i + 1]], "r:", linewidth=1.5)
        else:
            ax.plot([rri_time_s[i], rri_time_s[i + 1]],
                    [rri_ms[i], rri_ms[i + 1]], "b-.", linewidth=1.5)

    ax.set_ylabel("RRI (ms)")
    ax.set_title("(b) RR interval time series")
    ax.set_xlim([ecg_time_s[0], ecg_time_s[-1]])

    # --- (c) adRRI with threshold ---
    ax = axes[2]

    for i in range(len(adrri_ms)):
        color = "red" if flags_rri[i] else "blue"
        marker = "x" if flags_rri[i] else "D"
        ms = 8 if flags_rri[i] else 5
        ax.plot(adrri_time_s[i], adrri_ms[i], marker=marker, color=color,
                markersize=ms, markeredgewidth=1.5 if flags_rri[i] else 1)

    for i in range(len(adrri_ms) - 1):
        if flags_rri[i] or flags_rri[i + 1]:
            ax.plot([adrri_time_s[i], adrri_time_s[i + 1]],
                    [adrri_ms[i], adrri_ms[i + 1]], "r:", linewidth=1.5)
        else:
            ax.plot([adrri_time_s[i], adrri_time_s[i + 1]],
                    [adrri_ms[i], adrri_ms[i + 1]], "b-.", linewidth=1.5)

    ax.axhline(y=THETA_RPEAK, color="green", linestyle="--", linewidth=1.5,
               label=f"$\\theta$ = {THETA_RPEAK} ms")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("adRRI (ms)")
    ax.set_title("(c) Absolute difference of adjacent RR intervals")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim([ecg_time_s[0], ecg_time_s[-1]])

    plt.tight_layout()
    outpath = os.path.join(output_dir, "figure1_ecg_rri_adrri.png")
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    print(f"Saved Figure 1 to {outpath}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/")
    parser.add_argument("--output", default="outputs/")
    parser.add_argument("--patient", type=int, default=None)
    parser.add_argument("--epoch", type=int, default=None)
    args = parser.parse_args()
    make_figure1(args.data_dir, args.output, args.patient, args.epoch)
