# Original MATLAB Code

This directory contains the original MATLAB code for reference. The Python implementation in `adarri/` is a faithful port of this code.

## core/

Core pipeline files from the `Dennis_ECG_ArtifactReduction/` directory:

| File | Description |
|------|-------------|
| `fcn_pan_tompkin.m` | Pan-Tompkins R-peak detection algorithm |
| `fcn_get_rr_interval.m` | RR interval extraction from ECG |
| `fcnRealSampleData.m` | Load patient data and extract features |
| `fcnGetSensSpecAbsDif.m` | Sensitivity/specificity for adRRI threshold |
| `fcnGetSensSpecRRI.m` | Sensitivity/specificity for dual RRI thresholds |
| `a_Step4_KernelDensityPlotsDennis.m` | ROC curves and threshold optimization |
| `a_Step5_Gammadistribution_mbw.m` | Distribution visualization (Figure 2) |
| `a_Step6_PlotDetectedArtifacts.m` | Artifact visualization (Figure 1) |
| `fcnPlotKernelPDFestimatesAbsDif.m` | KDE plotting for adRRI |
| `fcnPlotKernelPDFestimatesRRI.m` | KDE plotting for RRI |
| `createPatches.m` | Bar plot utility with transparency |

## methods/

Comparison method implementations from the `CodeForSunil/` directory:

| File | Description |
|------|-------------|
| `fcn_Sunils_data.m` | Runs all 3 methods (A, B, C) on patient data |
| `fcn_calculate_absdiff.m` | Simple adRRI thresholding (Method A helper) |
| `fcn_clean_hrv4.m` | Clifford's 20% threshold method (Method C) |
| `fcn_Flag_Identification.m` | Flag identification procedure (START/LONG/SHORT BEAT) |

## Note on `fcn_Flag_Identification.m`

Line 7 of this file contains a bug: `if n<1` is always false since the loop starts at `n=1`. This means the SHORT BEAT / LONG BEAT logic (lines 8-36) never executes. The effective behavior is simple thresholding: `adRRI(n) > theta` → flag. The Python port (`adarri/detector.py`) provides both the bug-matching behavior (`flag_identification`) and the intended paper logic (`flag_identification_paper`).
