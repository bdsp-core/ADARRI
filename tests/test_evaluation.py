"""Tests for evaluation metrics."""

import numpy as np
import pytest
from adarri.evaluation import compute_metrics, balanced_bootstrap_metrics


class TestComputeMetrics:
    def test_perfect_classification(self):
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 0, 1, 1, 1])
        m = compute_metrics(y_true, y_pred)
        assert m["SE"] == 100.0
        assert m["SP"] == 100.0
        assert m["PPV"] == 100.0

    def test_all_wrong(self):
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([1, 1, 1, 0, 0, 0])
        m = compute_metrics(y_true, y_pred)
        assert m["SE"] == 0.0
        assert m["SP"] == 0.0
        assert m["PPV"] == 0.0

    def test_known_confusion_matrix(self):
        """TP=8, FP=2, FN=1, TN=9 -> SE=88.9%, SP=81.8%, PPV=80%"""
        y_true = np.array([1]*9 + [0]*11)
        y_pred = np.array([1]*8 + [0]*1 + [1]*2 + [0]*9)
        m = compute_metrics(y_true, y_pred)
        np.testing.assert_almost_equal(m["SE"], 88.89, decimal=1)
        np.testing.assert_almost_equal(m["SP"], 81.82, decimal=1)
        np.testing.assert_almost_equal(m["PPV"], 80.0, decimal=1)

    def test_likelihood_ratios(self):
        """LR+ = SE/(1-SP), LR- = (1-SE)/SP"""
        y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        y_pred = np.array([1, 1, 1, 0, 0, 0, 0, 1])
        m = compute_metrics(y_true, y_pred)
        se = m["SE"] / 100
        sp = m["SP"] / 100
        expected_lr_plus = se / (1 - sp)
        expected_lr_minus = (1 - se) / sp
        np.testing.assert_almost_equal(m["LR_plus"], expected_lr_plus, decimal=3)
        np.testing.assert_almost_equal(m["LR_minus"], expected_lr_minus, decimal=3)


class TestBalancedBootstrap:
    def test_balanced(self):
        """With already balanced data, should give similar results."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 1, 1, 1, 0])
        m = balanced_bootstrap_metrics(y_true, y_pred, n_iterations=20)
        assert 0 <= m["SE"] <= 100
        assert 0 <= m["SP"] <= 100
        assert 0 <= m["PPV"] <= 100

    def test_imbalanced(self):
        """With imbalanced data, bootstrap should balance classes."""
        y_true = np.array([0]*100 + [1]*10)
        y_pred = np.array([0]*100 + [1]*10)
        m = balanced_bootstrap_metrics(y_true, y_pred, n_iterations=20)
        assert m["SE"] == 100.0
        assert m["SP"] == 100.0
