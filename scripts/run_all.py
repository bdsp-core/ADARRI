#!/usr/bin/env python3
"""Master script: reproduce all results from the ADARRI paper.

Runs the full pipeline:
  1. Preprocess raw ECG data (if not already done)
  2. Generate Figure 1 (ECG, RRI, adRRI example)
  3. Generate Figure 2 (KDE distributions)
  4. Generate Figure 3 (ROC/PR curves, epoch level)
  5. Generate Figure 4 (ROC/PR curves, R-peak level)
  6. Generate Table 1 (overall performance)
  7. Generate Table 2 (per-subject performance)

Usage:
    python scripts/run_all.py --data-dir data/ --output outputs/
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main():
    parser = argparse.ArgumentParser(description="Reproduce all ADARRI paper results")
    parser.add_argument("--data-dir", default="data/",
                        help="Directory containing SegmentScores_Artifact_*.mat files")
    parser.add_argument("--output", default="outputs/",
                        help="Output directory for figures and tables")
    parser.add_argument("--skip-preprocess", action="store_true",
                        help="Skip preprocessing if features.pkl already exists")
    args = parser.parse_args()

    features_path = os.path.join(args.data_dir, "processed", "features.pkl")
    os.makedirs(args.output, exist_ok=True)

    # Step 1: Preprocess
    if not args.skip_preprocess or not os.path.exists(features_path):
        print("\n" + "=" * 60)
        print("STEP 1: Preprocessing ECG data")
        print("=" * 60)
        t0 = time.time()
        from scripts.preprocess_data import preprocess_all
        preprocess_all(args.data_dir, os.path.join(args.data_dir, "processed"))
        print(f"Preprocessing done in {time.time() - t0:.1f}s\n")
    else:
        print(f"Skipping preprocessing (found {features_path})")

    # Step 2: Figure 1
    print("\n" + "=" * 60)
    print("STEP 2: Generating Figure 1 (ECG, RRI, adRRI)")
    print("=" * 60)
    from scripts.figure1_ecg_rri_adrri import make_figure1
    make_figure1(args.data_dir, args.output)

    # Step 3: Figure 2
    print("\n" + "=" * 60)
    print("STEP 3: Generating Figure 2 (KDE distributions)")
    print("=" * 60)
    from scripts.figure2_kde_distributions import make_figure2
    make_figure2(features_path, args.output)

    # Step 4: Figure 3
    print("\n" + "=" * 60)
    print("STEP 4: Generating Figure 3 (ROC/PR, epoch level)")
    print("=" * 60)
    from scripts.figure3_roc_pr_epochs import make_figure3
    make_figure3(features_path, args.output)

    # Step 5: Table 1
    print("\n" + "=" * 60)
    print("STEP 5: Generating Table 1 (overall performance)")
    print("=" * 60)
    from scripts.table1_overall import compute_table1
    compute_table1(features_path, args.output)

    # Step 7: Table 2
    print("\n" + "=" * 60)
    print("STEP 7: Generating Table 2 (per-subject performance)")
    print("=" * 60)
    from scripts.table2_per_subject import compute_table2
    compute_table2(features_path, args.output)

    # Step 8: HRV Spectrogram
    ibi_path = os.path.join(args.data_dir, "ibi_for_Dennis.mat")
    if os.path.exists(ibi_path):
        print("\n" + "=" * 60)
        print("STEP 8: Generating HRV spectrogram (before/after artifact reduction)")
        print("=" * 60)
        from scripts.figure_hrv_spectrogram import make_figure
        make_figure(ibi_path, args.output)
    else:
        print(f"\nSkipping HRV spectrogram (ibi_for_Dennis.mat not found in {args.data_dir})")

    print("\n" + "=" * 60)
    print("ALL DONE! Results saved to: " + args.output)
    print("=" * 60)


if __name__ == "__main__":
    main()
