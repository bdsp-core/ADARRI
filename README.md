# ADARRI: Detecting Spurious R-Peaks in ECG for HRV Analysis in the ICU

Python implementation of the ADARRI algorithm for automatically detecting artifacts in R-peak detections from electrocardiogram (ECG) recordings, designed for heart rate variability (HRV) analysis in intensive care unit (ICU) patients.

## Citation

If you use this code, please cite:

> Rebergen DJ, Nagaraj SB, Rosenthal ES, Bianchi MT, van Putten MJAM, Westover MB.
> **ADARRI: a novel method to detect spurious R-peaks in the electrocardiogram for heart rate variability analysis in the intensive care unit.**
> *J Clin Monit Comput.* 2018;32:53-61.
> [doi:10.1007/s10877-017-9999-9](https://doi.org/10.1007/s10877-017-9999-9)

## Overview

ADARRI uses the **absolute difference of adjacent RR intervals (adRRI)** to identify artifacts in R-peak detections. The key insight is that sudden changes in RR intervals (measured by adRRI) reliably distinguish true R-peak detections from spurious ones using a single, patient-independent threshold.

**Key results from the paper:**
- Epoch-level: SE=96%, SP=83%, PPV=85% at threshold theta=276 ms
- Individual R-peak level: SE=99%, SP=95%, PPV=95% at threshold theta=85 ms
- Outperforms Berntson's (Method B) and Clifford's (Method C) methods on ICU data

## Installation

```bash
git clone https://github.com/bdsp-core/ADARRI.git
cd ADARRI
pip install -e .
```

## Quick Start: Using ADARRI on New ECG Data

```python
import numpy as np
from adarri.peak_detection import detect_r_peaks
from adarri.rri import compute_rri, compute_adrri
from adarri.detector import flag_identification, THETA_RPEAK

# Load your ECG signal (240 Hz expected, adjust sampling_rate as needed)
ecg_signal = np.load("your_ecg_data.npy")
sampling_rate = 240  # Hz

# Step 1: Detect R-peaks
r_peaks = detect_r_peaks(ecg_signal, sampling_rate)

# Step 2: Compute RR intervals
rri, times = compute_rri(r_peaks, sampling_rate)
rri_ms = rri * 1000  # Convert to milliseconds

# Step 3: Compute adRRI with leading zero (matching paper convention)
adrri = np.concatenate([[0], np.abs(np.diff(rri_ms))])

# Step 4: Flag artifacts using ADARRI threshold (85 ms for individual R-peaks)
flags = flag_identification(rri_ms, adrri, theta=THETA_RPEAK)

# Results
clean_peaks = r_peaks[~flags[:len(r_peaks)]]
artifact_peaks = r_peaks[flags[:len(r_peaks)]]
print(f"Total R-peaks: {len(r_peaks)}")
print(f"Clean: {len(clean_peaks)}, Artifacts: {len(artifact_peaks)}")
```

## Reproducing Paper Results

### 1. Obtain the data

Place the `SegmentScores_Artifact_*.mat` files in the `data/` directory. See [data/README.md](data/README.md) for details on the expected data format.

### 2. Run the full pipeline

```bash
# Reproduce all figures and tables
python scripts/run_all.py --data-dir data/ --output outputs/
```

Or run individual steps:

```bash
# Step 1: Preprocess raw ECG data
python scripts/preprocess_data.py --data-dir data/ --output data/processed/

# Step 2: Generate individual figures/tables
python scripts/figure1_ecg_rri_adrri.py --data-dir data/ --output outputs/
python scripts/figure2_kde_distributions.py --features data/processed/features.pkl --output outputs/
python scripts/figure3_roc_pr_epochs.py --features data/processed/features.pkl --output outputs/
python scripts/figure4_roc_pr_peaks.py --features data/processed/features.pkl --output outputs/
python scripts/table1_overall.py --features data/processed/features.pkl --output outputs/
python scripts/table2_per_subject.py --features data/processed/features.pkl --output outputs/
```

### Outputs

| Script | Paper Element | Description |
|--------|--------------|-------------|
| `figure1_ecg_rri_adrri.py` | Figure 1 | ECG signal, RRI, and adRRI with R-peak labels |
| `figure2_kde_distributions.py` | Figure 2 | Kernel density of log(adRRI) for valid vs artifact |
| `figure3_roc_pr_epochs.py` | Figure 3 | ROC and PR curves for epoch-level evaluation |
| `figure4_roc_pr_peaks.py` | Figure 4 | ROC and PR curves for individual R-peak evaluation |
| `table1_overall.py` | Table 1 | All-subject performance (SE, SP, PPV, LR+, LR-) |
| `table2_per_subject.py` | Table 2 | Per-subject medians and IQR |

## Methods

The code implements three artifact detection methods compared in the paper:

- **Method A (ADARRI)**: Threshold on the absolute difference of adjacent RR intervals. Uses a single, patient-independent threshold derived from the empirical adRRI distribution.
- **Method B (Berntson)**: IQR/MAD-based thresholding from Berntson et al. (1990). Patient-specific threshold computed from the interquartile range and median of the adRRI distribution.
- **Method C (Clifford)**: Percentage-based threshold from Clifford et al. (2002). Flags beats where the RR interval deviates >20% from the previous valid interval.

## Repository Structure

```
ADARRI/
├── adarri/              # Python package
│   ├── io.py            # Data loading (.mat files)
│   ├── peak_detection.py # Pan-Tompkins R-peak detection
│   ├── rri.py           # RRI and adRRI computation
│   ├── detector.py      # ADARRI method (Method A)
│   ├── berntson.py      # Berntson method (Method B)
│   ├── clifford.py      # Clifford method (Method C)
│   └── evaluation.py    # Performance metrics
├── scripts/             # Reproduce paper results
├── matlab/              # Original MATLAB code (reference)
├── tests/               # Unit tests
├── data/                # Data directory (see data/README.md)
├── outputs/             # Generated figures and tables
└── paper/               # Published paper PDF
```

## Notes on Reproducing Results

The Python implementation includes a faithful port of the MATLAB Pan-Tompkins R-peak detection algorithm (`fcn_pan_tompkin.m`), including adaptive thresholding, search-back for missed QRS complexes, T-wave discrimination, and dual verification on the bandpass filtered signal. This produces results that closely match the paper:

- **Epoch-level (Method A)**: Our optimal threshold is 293 ms (paper: 276 ms), with SE=95%, SP=85%, PPV=86% (paper: SE=96%, SP=83%, PPV=85%). The ROC curve (AUC=0.952) confirms excellent discrimination. Per-subject medians are SE=98%, SP=93% (paper: SE=99%, SP=93%).
- **R-peak level (Figure 4)**: The paper's R-peak evaluation (SE=99%, SP=95%, PPV=95% at theta=85 ms) used per-R-peak expert annotations stored in a file called `measures.mat`, which contained the variables `True_Rpeak` (ground truth) and `ADARRI_Rpeak` (method predictions) along with epoch-level equivalents (`True_epoch`, `ADARRI_epoch`). The MATLAB script `Manuscript/calculations/a_Calculate_performances.m` computed performance metrics from this file using balanced subsampling (`fcn_get_data_balanced.m`). **This file has been lost** — it was likely on the original author's local machine and was never uploaded to the shared data repository. Without per-R-peak expert annotations, the Figure 4 ROC/PR curves cannot be faithfully reproduced; our epoch-inherited labels (all peaks in an artifact epoch labeled as artifact) give ~50/50 class balance instead of the paper's 91.5%/8.5% split, producing poor results. If `measures.mat` is recovered, the R-peak level evaluation can be completed using the existing `CodeForSunil/` MATLAB functions (`fcn_Sunils_data.m`, `fcn_Flag_Identification.m`, `fcn_calculate_absdiff.m`), which apply Methods A/B/C at the individual R-peak level.
- **Methods B and C**: Berntson (Method B) shows low specificity because its per-epoch IQR/MAD threshold is often too permissive. Clifford (Method C) achieves SE=98%, SP=58%. These results support the paper's conclusion that ADARRI is more robust.

The original MATLAB code is included in `matlab/` for reference.

## Testing

```bash
pytest tests/ -v
```

## Dependencies

- Python >= 3.8
- numpy, scipy, matplotlib, scikit-learn, pandas, h5py

## License

MIT License. See [LICENSE](LICENSE) for details.
