"""Data loading utilities for ADARRI.

Loads MATLAB .mat files containing ECG epochs and pre-computed features.
"""

import glob
import os

import numpy as np
import scipy.io


def load_segment_scores(filepath):
    """Load a SegmentScores_Artifact_*.mat file.

    Each file contains ECG epochs for one patient:
      - ECG0: cell array of valid (clean) epochs (240 Hz ECG signals)
      - ECG1: cell array of artifact-containing epochs

    Args:
        filepath: Path to the .mat file.

    Returns:
        dict with keys:
            'valid_epochs': list of 1D numpy arrays (clean ECG signals)
            'artifact_epochs': list of 1D numpy arrays (artifact ECG signals)
    """
    mat = scipy.io.loadmat(filepath, squeeze_me=True)

    valid = _cell_to_list(mat["ECG0"], max_epochs=60)
    artifact = _cell_to_list(mat["ECG1"], max_epochs=60)

    return {"valid_epochs": valid, "artifact_epochs": artifact}


def load_all_patients(data_dir):
    """Load all SegmentScores_Artifact_*.mat files from a directory.

    Args:
        data_dir: Path to the directory containing .mat files.

    Returns:
        list of dicts, each from load_segment_scores(), with added 'patient_id' key.
    """
    # Support both naming conventions: patient_*.mat and SegmentScores_Artifact_*.mat
    files = sorted(glob.glob(os.path.join(data_dir, "patient_*.mat")))
    if not files:
        files = sorted(glob.glob(os.path.join(data_dir, "SegmentScores_Artifact_*.mat")))
    if not files:
        raise FileNotFoundError(
            f"No patient_*.mat or SegmentScores_Artifact_*.mat files found in {data_dir}"
        )

    patients = []
    for f in files:
        patient_id = os.path.basename(f).replace("SegmentScores_Artifact_", "").replace("patient_", "").replace(".mat", "")
        data = load_segment_scores(f)
        data["patient_id"] = patient_id
        data["filepath"] = f
        patients.append(data)

    return patients


def load_xy_realdata(filepath):
    """Load XY_RealData.mat containing pre-computed features.

    This is a large file (~428MB) that may be in v7.3 (HDF5) format.

    Args:
        filepath: Path to XY_RealData.mat.

    Returns:
        dict with keys 'X' and 'Y'.
    """
    try:
        mat = scipy.io.loadmat(filepath, squeeze_me=True)
        return {"X": mat["X"], "Y": mat["Y"]}
    except NotImplementedError:
        # v7.3 HDF5 format
        import h5py

        with h5py.File(filepath, "r") as f:
            return {"X": np.array(f["X"]), "Y": np.array(f["Y"])}


def load_thetas(filepath):
    """Load THETAS.mat containing per-patient optimal thresholds.

    Args:
        filepath: Path to THETAS.mat.

    Returns:
        numpy array of shape (n_patients, 3) with columns [th0, th1, th_adrri].
    """
    mat = scipy.io.loadmat(filepath, squeeze_me=True)
    return mat["THETAS"]


def _cell_to_list(cell_array, max_epochs=60):
    """Convert a MATLAB cell array to a list of numpy arrays.

    Args:
        cell_array: numpy object array from scipy.io.loadmat.
        max_epochs: Maximum number of epochs to keep.

    Returns:
        List of 1D numpy arrays.
    """
    if cell_array.ndim == 0:
        # Single element
        return [np.asarray(cell_array.item(), dtype=np.float64)]

    result = []
    items = cell_array.flat if cell_array.ndim > 1 else cell_array
    for item in items:
        if item is not None and np.asarray(item).size > 0:
            result.append(np.asarray(item, dtype=np.float64).ravel())
        if len(result) >= max_epochs:
            break
    return result
