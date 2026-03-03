"""Performance evaluation metrics for artifact detection.

Computes sensitivity, specificity, PPV, likelihood ratios, and provides
balanced bootstrap evaluation matching the paper's methodology.
"""

import numpy as np
import pandas as pd


def compute_metrics(y_true, y_pred):
    """Compute SE, SP, PPV, LR+, LR- from binary labels.

    Following Equations (3) and (4) from the paper:
        LR+ = SE / (1 - SP)
        LR- = (1 - SE) / SP

    Args:
        y_true: Ground truth (1=artifact, 0=valid).
        y_pred: Predictions (1=artifact, 0=valid).

    Returns:
        dict with keys: SE, SP, PPV, LR_plus, LR_minus (percentages for SE/SP/PPV).
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    se = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    sp = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    lr_plus = se / (1.0 - sp) if sp < 1.0 else float("inf")
    lr_minus = (1.0 - se) / sp if sp > 0 else float("inf")

    return {
        "SE": se * 100,
        "SP": sp * 100,
        "PPV": ppv * 100,
        "LR_plus": lr_plus,
        "LR_minus": lr_minus,
    }


def balanced_bootstrap_metrics(y_true, y_pred, n_iterations=20, random_state=42):
    """Compute metrics using balanced bootstrap (Section 2.6 of paper).

    'We generated balanced sets by choosing randomly without replacement
    from the larger group to yield groups of equal size. This balancing
    procedure was repeated 20 times and the average values were used.'

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        n_iterations: Number of bootstrap iterations (default 20).
        random_state: Random seed for reproducibility.

    Returns:
        dict with averaged metrics.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    rng = np.random.RandomState(random_state)

    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]

    n_min = min(len(pos_idx), len(neg_idx))
    if n_min == 0:
        return compute_metrics(y_true, y_pred)

    all_metrics = []
    for _ in range(n_iterations):
        if len(pos_idx) > len(neg_idx):
            sampled_pos = rng.choice(pos_idx, n_min, replace=False)
            sampled_neg = neg_idx
        else:
            sampled_pos = pos_idx
            sampled_neg = rng.choice(neg_idx, n_min, replace=False)

        idx = np.concatenate([sampled_pos, sampled_neg])
        metrics = compute_metrics(y_true[idx], y_pred[idx])
        all_metrics.append(metrics)

    return {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}


def per_subject_summary(subject_metrics):
    """Compute median (IQR) summary across subjects (Table 2).

    Args:
        subject_metrics: List of metric dicts (one per subject).

    Returns:
        DataFrame with median and IQR for each metric.
    """
    df = pd.DataFrame(subject_metrics)
    summary = {}
    for col in df.columns:
        med = df[col].median()
        q25 = df[col].quantile(0.25)
        q75 = df[col].quantile(0.75)
        summary[col] = {"median": med, "q25": q25, "q75": q75}

    return pd.DataFrame(summary).T


def format_table_row(metrics, method_name):
    """Format a metrics dict as a table row string.

    Args:
        metrics: Dict from compute_metrics or balanced_bootstrap_metrics.
        method_name: Name of the method.

    Returns:
        Formatted string.
    """
    return (
        f"{method_name:<20s} "
        f"SE={metrics['SE']:.0f}%  "
        f"SP={metrics['SP']:.0f}%  "
        f"PPV={metrics['PPV']:.0f}%  "
        f"LR+={metrics['LR_plus']:.1f}  "
        f"LR-={metrics['LR_minus']:.3f}"
    )
