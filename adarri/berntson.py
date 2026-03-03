"""Berntson's artifact detection method (Method B).

Implements the IQR/MAD-based threshold from:
    Berntson GG, Quigley KS, Jang JF, Boysen ST. An approach to artifact
    identification: application to heart period data. Psychophysiology.
    1990;27(5):586-598.

Ported from fcn_Sunils_data.m, lines 10-13.
"""

import numpy as np
from scipy.stats import iqr as scipy_iqr


def berntson_threshold(adrri):
    """Compute Berntson's threshold from adRRI distribution.

    From the MATLAB code:
        MEDn = iqr(adRRI)/2 * 3.32
        MADa = (median(adRRI) - 2.9*iqr(adRRI)) / 3
        threshold = (abs(MEDn) + abs(MADa)) / 2

    This is a patient-specific threshold derived from the IQR and
    median of the adRRI distribution.

    Args:
        adrri: Array of adRRI values (in ms).

    Returns:
        threshold: The Berntson detection threshold (in ms).
    """
    iqr_val = scipy_iqr(adrri)
    med_val = np.median(adrri)

    MEDn = iqr_val / 2.0 * 3.32
    MADa = (med_val - 2.9 * iqr_val) / 3.0
    threshold = (abs(MEDn) + abs(MADa)) / 2.0

    return threshold


def detect_artifacts_berntson(rri, adrri):
    """Detect artifacts using Berntson's method (Method B).

    Computes a patient-specific threshold from the adRRI distribution,
    then applies flag identification.

    Args:
        rri: Array of RR intervals (in ms).
        adrri: Array of adRRI values (in ms), with leading 0
               (i.e., [0, abs(diff(rri))]).

    Returns:
        flags: Boolean array of length len(adrri). True = artifact.
    """
    from .detector import flag_identification

    threshold = berntson_threshold(adrri)
    flags = flag_identification(rri, adrri, threshold)
    return flags


def classify_epoch_berntson(rri_epochs, adrri_epochs, labels):
    """Classify epochs using Berntson's method for evaluation.

    Computes a single threshold from all data, then classifies each epoch.

    Args:
        rri_epochs: List of RRI arrays (one per epoch).
        adrri_epochs: List of adRRI arrays (one per epoch).
        labels: Ground truth labels (0=valid, 1=artifact).

    Returns:
        predictions: Array of predicted labels.
        threshold: The computed Berntson threshold.
    """
    # Pool all adRRI data to compute threshold
    all_adrri = np.concatenate(adrri_epochs)
    threshold = berntson_threshold(all_adrri)

    predictions = np.zeros(len(labels), dtype=int)
    for j, adrri in enumerate(adrri_epochs):
        if np.max(adrri) > threshold:
            predictions[j] = 1

    return predictions, threshold
