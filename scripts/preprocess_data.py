#!/usr/bin/env python3
"""Step 1: Extract features from raw ECG .mat files.

Loads all SegmentScores_Artifact_*.mat files, runs Pan-Tompkins R-peak
detection, computes RRI and adRRI, and saves processed features.

This replaces the need for the pre-computed XY_RealData.mat file.

Usage:
    python scripts/preprocess_data.py --data-dir data/ --output data/processed/
"""

import argparse
import os
import pickle
import sys

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from adarri.io import load_all_patients
from adarri.rri import process_epoch


def preprocess_all(data_dir, output_dir):
    """Process all patients and save features."""
    os.makedirs(output_dir, exist_ok=True)

    patients = load_all_patients(data_dir)
    print(f"Found {len(patients)} patients")

    all_X = []  # Per-patient feature data
    all_Y = []  # Per-patient labels

    for pt_idx, patient in enumerate(tqdm(patients, desc="Processing patients")):
        patient_id = patient["patient_id"]
        valid_epochs = patient["valid_epochs"]
        artifact_epochs = patient["artifact_epochs"]

        labels = np.concatenate([
            np.zeros(len(valid_epochs), dtype=int),
            np.ones(len(artifact_epochs), dtype=int),
        ])

        epoch_data = []
        valid_labels = []

        # Process valid epochs
        for i, ecg in enumerate(valid_epochs):
            try:
                result = process_epoch(ecg, sampling_rate=240, step_ms=10)
                if result is not None:
                    epoch_data.append({
                        "rri_ms": result["rri_ms"],
                        "adrri_ms": result["adrri_ms"],
                        "r_peaks": result["r_peaks"],
                        "rri_raw": result["rri_raw"],
                        "times_raw": result["times_raw"],
                    })
                    valid_labels.append(0)
            except Exception as e:
                print(f"  Patient {patient_id}, valid epoch {i}: {e}")

        # Process artifact epochs
        for i, ecg in enumerate(artifact_epochs):
            try:
                result = process_epoch(ecg, sampling_rate=240, step_ms=10)
                if result is not None:
                    epoch_data.append({
                        "rri_ms": result["rri_ms"],
                        "adrri_ms": result["adrri_ms"],
                        "r_peaks": result["r_peaks"],
                        "rri_raw": result["rri_raw"],
                        "times_raw": result["times_raw"],
                    })
                    valid_labels.append(1)
            except Exception as e:
                print(f"  Patient {patient_id}, artifact epoch {i}: {e}")

        all_X.append(epoch_data)
        all_Y.append(np.array(valid_labels))

        print(f"  Patient {patient_id}: {sum(1 for l in valid_labels if l==0)} valid, "
              f"{sum(1 for l in valid_labels if l==1)} artifact epochs")

    # Save processed data
    output_path = os.path.join(output_dir, "features.pkl")
    with open(output_path, "wb") as f:
        pickle.dump({
            "X": all_X,
            "Y": all_Y,
            "patient_ids": [p["patient_id"] for p in patients],
        }, f)

    print(f"\nSaved processed features to {output_path}")
    print(f"Total patients: {len(all_X)}")
    total_epochs = sum(len(y) for y in all_Y)
    total_valid = sum(np.sum(y == 0) for y in all_Y)
    total_artifact = sum(np.sum(y == 1) for y in all_Y)
    print(f"Total epochs: {total_epochs} ({total_valid} valid, {total_artifact} artifact)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess ECG data")
    parser.add_argument("--data-dir", default="data/", help="Directory with .mat files")
    parser.add_argument("--output", default="data/processed/", help="Output directory")
    args = parser.parse_args()

    preprocess_data_dir = args.data_dir
    preprocess_all(preprocess_data_dir, args.output)
