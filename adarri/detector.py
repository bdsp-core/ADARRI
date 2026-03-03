"""ADARRI artifact detection (Method A).

Implements the core ADARRI algorithm:
  - Epoch-level: classify epochs as artifact if max(adRRI) or RRI exceeds threshold
  - R-peak-level: flag individual R-peaks using the flag identification procedure

Reference thresholds from the paper:
  - Epoch-level: theta = 276 ms (adRRI)
  - Individual R-peak: theta = 85 ms (adRRI)
"""

import numpy as np

# Optimal thresholds from Table 1 of the paper
THETA_EPOCH = 276.0  # ms, for epoch-level detection
THETA_RPEAK = 85.0   # ms, for individual R-peak detection


def classify_epoch_adrri(adrri, theta):
    """Classify an epoch using the adRRI threshold (log-space).

    Mirrors fcnGetSensSpecAbsDif.m: artifact if max(data) > exp(theta),
    where theta is in log-space and data is the raw adRRI.

    Args:
        adrri: Array of adRRI values for one epoch (in ms).
        theta: Threshold in log-space (compared as exp(theta) in ms-space).

    Returns:
        True if epoch is classified as artifact.
    """
    if len(adrri) == 0:
        return False
    return float(np.max(adrri)) > np.exp(theta)


def classify_epoch_rri(rri, theta0, theta1):
    """Classify an epoch using dual RRI thresholds (log-space).

    Mirrors fcnGetSensSpecRRI.m: artifact if min(RRI) < exp(theta0)
    OR max(RRI) > exp(theta1).

    Args:
        rri: Array of RRI values for one epoch (in ms).
        theta0: Lower threshold in log-space.
        theta1: Upper threshold in log-space.

    Returns:
        True if epoch is classified as artifact.
    """
    if len(rri) == 0:
        return False
    return float(np.min(rri)) < np.exp(theta0) or float(np.max(rri)) > np.exp(theta1)


def flag_identification(rri, adrri, theta):
    """Flag individual R-peaks as artifacts using adRRI threshold.

    This implements the FLAG identification procedure from the paper
    (Figures 5-7: START, LONG BEAT, SHORT BEAT). Note: the original
    MATLAB implementation (fcn_Flag_Identification.m) has a bug on line 7
    (``if n<1`` is always false since the loop starts at n=1), so the
    SHORT BEAT / LONG BEAT logic never executes. The effective behavior
    is simple thresholding: adRRI(n) > theta -> flag.

    This implementation provides the simple (bug-matching) behavior by
    default, with an option to enable the intended paper logic.

    Args:
        rri: Array of RR intervals (in ms). Length N.
        adrri: Array of adRRI values (in ms). Length N (with leading 0,
               matching MATLAB: adRRI = [0 abs(diff(interval))]).
        theta: Threshold in ms.

    Returns:
        flags: Boolean array of length N. True = artifact.
    """
    n = len(adrri)
    flags = np.zeros(n, dtype=bool)

    for i in range(n - 2):
        if adrri[i] > theta:
            flags[i] = True

    return flags


def flag_identification_paper(rri, adrri, theta):
    """Flag identification with the intended paper logic (Figures 5-7).

    This implements what the code was meant to do (fixing the bug):
      - START: if adRRI > theta, activate flag
      - SHORT BEAT: if RRI decreased (spurious peak), check if removing
        the peak would resolve the artifact
      - LONG BEAT: if RRI increased (missed peak), check if splitting
        the interval would resolve the artifact

    Args:
        rri: Array of RR intervals (in ms). Length N.
        adrri: Array of adRRI values (in ms). Length N (with leading 0).
        theta: Threshold in ms.

    Returns:
        flags: Boolean array of length N. True = artifact.
    """
    n = len(adrri)
    flags = np.zeros(n, dtype=bool)

    for i in range(1, n - 2):
        if adrri[i] > theta:
            flags[i] = True

            if adrri[i - 1] < theta:
                if rri[i] - rri[i + 1] < 0:
                    exit_code = 2  # LONG BEAT
                else:
                    exit_code = 1  # SHORT BEAT
            else:
                # Can't evaluate (previous also flagged)
                flags[i] = True
                continue

            if exit_code == 1:  # SHORT BEAT
                if i + 2 < n and adrri[i + 2] < theta:
                    if rri[i - 1] < rri[i + 1]:
                        x = rri[i - 1] + rri[i]
                    else:
                        x = rri[i + 1] + rri[i]
                    if x - rri[i - 1] > theta and x - rri[i + 1] > theta:
                        flags[i] = False  # FLAG OFF
                    else:
                        flags[i] = True   # FLAG ON
                else:
                    flags[i] = True  # Can't evaluate

            elif exit_code == 2:  # LONG BEAT
                if i + 2 < n and adrri[i + 2] < theta:
                    x = rri[i] / 2.0
                    if x - rri[i - 1] < -theta and x - rri[i + 1] < -theta:
                        flags[i] = False  # FLAG OFF
                    else:
                        flags[i] = True   # FLAG ON
                else:
                    flags[i] = True  # Can't evaluate

    return flags


def find_optimal_thresholds_adrri(X, Y, n_thresholds=500):
    """Find the optimal adRRI threshold by sweeping and maximizing accuracy.

    Mirrors the abs(diff(RRI)) portion of a_Step4_KernelDensityPlotsDennis.m.

    Args:
        X: List (per epoch) of dicts or arrays. Each X[j] should support
           X[j]['adrri_ms'] (or be passed as raw adRRI arrays).
        Y: Array of labels (0=valid, 1=artifact) for each epoch.
        n_thresholds: Number of threshold values to sweep.

    Returns:
        best_theta: Optimal threshold (in log-space).
        thresholds: Array of swept thresholds.
        se_arr: Sensitivity at each threshold.
        sp_arr: Specificity at each threshold.
        acc_arr: Accuracy at each threshold.
    """
    # Collect all adRRI data to determine range
    all_log_adrri = []
    for j in range(len(X)):
        data = X[j]
        if isinstance(data, dict):
            vals = data["adrri_ms"]
        else:
            vals = data
        log_vals = np.log(np.maximum(vals, 1e-10))
        all_log_adrri.extend(log_vals.tolist())

    all_log_adrri = np.array(all_log_adrri)
    th_min, th_max = np.min(all_log_adrri), np.max(all_log_adrri)
    thresholds = np.linspace(th_min, th_max, n_thresholds)

    Y = np.asarray(Y)
    se_arr = np.zeros(n_thresholds)
    sp_arr = np.zeros(n_thresholds)
    acc_arr = np.zeros(n_thresholds)

    for t_idx, th in enumerate(thresholds):
        yh = np.zeros(len(Y), dtype=int)
        for j in range(len(X)):
            data = X[j]
            if isinstance(data, dict):
                vals = data["adrri_ms"]
            else:
                vals = data
            if np.max(vals) > np.exp(th):
                yh[j] = 1

        n_pos = np.sum(Y == 1)
        n_neg = np.sum(Y == 0)
        se_arr[t_idx] = np.sum((Y == 1) & (yh == 1)) / n_pos if n_pos > 0 else 0
        sp_arr[t_idx] = np.sum((Y == 0) & (yh == 0)) / n_neg if n_neg > 0 else 0
        acc_arr[t_idx] = np.sum(Y == yh) / len(Y)

    best_idx = np.argmax(acc_arr)
    return thresholds[best_idx], thresholds, se_arr, sp_arr, acc_arr


def find_optimal_thresholds_rri(X, Y, n_thresholds=500):
    """Find optimal dual RRI thresholds by sweeping and maximizing accuracy.

    Mirrors the RRI portion of a_Step4_KernelDensityPlotsDennis.m.

    Args:
        X: List (per epoch) of dicts or arrays with RRI data.
        Y: Array of labels (0=valid, 1=artifact).
        n_thresholds: Number of threshold values per dimension.

    Returns:
        best_theta0: Optimal lower threshold (log-space).
        best_theta1: Optimal upper threshold (log-space).
        thresholds: Array of swept thresholds.
        se_mat: 2D sensitivity matrix.
        sp_mat: 2D specificity matrix.
        acc_mat: 2D accuracy matrix.
    """
    # Collect all RRI data to determine range
    all_log_rri = []
    for j in range(len(X)):
        data = X[j]
        if isinstance(data, dict):
            vals = data["rri_ms"]
        else:
            vals = data
        log_vals = np.log(np.maximum(vals, 1e-10))
        all_log_rri.extend(log_vals.tolist())

    all_log_rri = np.array(all_log_rri)
    th_min, th_max = np.min(all_log_rri), np.max(all_log_rri)
    thresholds = np.linspace(th_min, th_max, n_thresholds)

    Y = np.asarray(Y)
    se_mat = np.zeros((n_thresholds, n_thresholds))
    sp_mat = np.zeros((n_thresholds, n_thresholds))
    acc_mat = np.zeros((n_thresholds, n_thresholds))

    for i in range(n_thresholds):
        for k in range(i, n_thresholds):
            yh = np.zeros(len(Y), dtype=int)
            for j in range(len(X)):
                data = X[j]
                if isinstance(data, dict):
                    vals = data["rri_ms"]
                else:
                    vals = data
                if np.min(vals) < np.exp(thresholds[i]):
                    yh[j] = 1
                elif np.max(vals) > np.exp(thresholds[k]):
                    yh[j] = 1

            n_pos = np.sum(Y == 1)
            n_neg = np.sum(Y == 0)
            se_mat[i, k] = np.sum((Y == 1) & (yh == 1)) / n_pos if n_pos > 0 else 0
            sp_mat[i, k] = np.sum((Y == 0) & (yh == 0)) / n_neg if n_neg > 0 else 0
            acc_mat[i, k] = np.sum(Y == yh) / len(Y)

    best_idx = np.unravel_index(np.argmax(acc_mat), acc_mat.shape)
    best_theta0 = thresholds[best_idx[0]]
    best_theta1 = thresholds[best_idx[1]]

    return best_theta0, best_theta1, thresholds, se_mat, sp_mat, acc_mat
