"""Tests for ADARRI artifact detection."""

import numpy as np
import pytest
from adarri.detector import (
    classify_epoch_adrri,
    classify_epoch_rri,
    flag_identification,
    flag_identification_paper,
    THETA_EPOCH,
    THETA_RPEAK,
)


class TestClassifyEpochAdRRI:
    def test_clean_epoch(self):
        """Small adRRI values should not trigger artifact flag."""
        adrri = np.array([10, 5, 8, 3, 12])
        # theta in log space, exp(theta) >> max(adrri)
        assert not classify_epoch_adrri(adrri, theta=np.log(1000))

    def test_artifact_epoch(self):
        """Large adRRI should trigger artifact flag."""
        adrri = np.array([10, 5, 500, 3, 12])
        assert classify_epoch_adrri(adrri, theta=np.log(100))

    def test_boundary(self):
        """Value exactly at threshold."""
        adrri = np.array([100.0])
        # exp(log(100)) = 100, max(adrri) = 100, NOT > 100
        assert not classify_epoch_adrri(adrri, theta=np.log(100))
        # Slightly below threshold
        assert classify_epoch_adrri(adrri, theta=np.log(99))

    def test_empty(self):
        assert not classify_epoch_adrri(np.array([]), theta=5)


class TestClassifyEpochRRI:
    def test_normal_rri(self):
        """Normal RRI range should not trigger."""
        rri = np.array([800, 900, 850, 950])
        # Thresholds far outside normal range
        assert not classify_epoch_rri(rri, theta0=np.log(200), theta1=np.log(2000))

    def test_too_short_rri(self):
        """Very short RRI should trigger (lower bound)."""
        rri = np.array([800, 100, 850])  # 100 ms is abnormally short
        assert classify_epoch_rri(rri, theta0=np.log(300), theta1=np.log(2000))

    def test_too_long_rri(self):
        """Very long RRI should trigger (upper bound)."""
        rri = np.array([800, 3000, 850])  # 3000 ms is abnormally long
        assert classify_epoch_rri(rri, theta0=np.log(200), theta1=np.log(2000))


class TestFlagIdentification:
    def test_clean_signal(self):
        """No flags for clean signal."""
        rri = np.array([1000, 1000, 1000, 1000])
        adrri = np.array([0, 0, 0, 0])
        flags = flag_identification(rri, adrri, theta=50)
        assert not np.any(flags)

    def test_single_artifact(self):
        """Single large adRRI should be flagged."""
        rri = np.array([1000, 500, 1000, 1000, 1000])
        adrri = np.array([0, 500, 500, 0, 0])
        flags = flag_identification(rri, adrri, theta=100)
        assert flags[1]
        assert flags[2]


class TestFlagIdentificationPaper:
    def test_clean_signal(self):
        """No flags for clean signal."""
        rri = np.array([1000, 1000, 1000, 1000])
        adrri = np.array([0, 0, 0, 0])
        flags = flag_identification_paper(rri, adrri, theta=50)
        assert not np.any(flags)


class TestThresholdConstants:
    def test_paper_values(self):
        """Verify threshold constants match the paper."""
        assert THETA_EPOCH == 276.0
        assert THETA_RPEAK == 85.0
