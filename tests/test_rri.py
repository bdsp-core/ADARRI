"""Tests for RRI and adRRI computation."""

import numpy as np
import pytest
from adarri.rri import compute_rri, compute_adrri, interpolate_rri


class TestComputeRRI:
    def test_constant_heart_rate(self):
        """R-peaks at constant intervals should give constant RRI."""
        fs = 240
        # Peaks every 240 samples = 1 second = 1000 ms
        r_peaks = np.array([0, 240, 480, 720, 960])
        rri, times = compute_rri(r_peaks, fs)
        np.testing.assert_allclose(rri, [1.0, 1.0, 1.0, 1.0])  # seconds
        np.testing.assert_allclose(times, [0, 1.0, 2.0, 3.0])

    def test_variable_heart_rate(self):
        """Variable spacing should give variable RRI."""
        fs = 240
        r_peaks = np.array([0, 200, 500, 720])
        rri, times = compute_rri(r_peaks, fs)
        expected_rri = np.diff(r_peaks) / fs
        np.testing.assert_allclose(rri, expected_rri)

    def test_length(self):
        """RRI should have length n-1 for n peaks."""
        r_peaks = np.array([10, 100, 250, 400, 600])
        rri, times = compute_rri(r_peaks, 240)
        assert len(rri) == len(r_peaks) - 1
        assert len(times) == len(r_peaks) - 1


class TestComputeAdRRI:
    def test_constant_rri(self):
        """Constant RRI should give zero adRRI."""
        rri = np.array([1000, 1000, 1000, 1000])
        adrri = compute_adrri(rri)
        np.testing.assert_allclose(adrri, [0, 0, 0])

    def test_known_values(self):
        """Test with known values."""
        rri = np.array([1000, 800, 1200, 900])
        adrri = compute_adrri(rri)
        expected = np.array([200, 400, 300])
        np.testing.assert_allclose(adrri, expected)

    def test_length(self):
        """adRRI should have length n-1 for n RRI values."""
        rri = np.array([1000, 900, 1100, 950, 1050])
        adrri = compute_adrri(rri)
        assert len(adrri) == len(rri) - 1


class TestInterpolateRRI:
    def test_preserves_endpoints(self):
        """Interpolated values at original points should match."""
        times = np.array([0, 1.0, 2.0, 3.0])
        rri = np.array([1000, 950, 1050, 980])
        rri_interp, times_interp = interpolate_rri(rri, times, step_s=1.0)
        # At original time points, should be close to original values
        np.testing.assert_allclose(rri_interp[0], rri[0], rtol=0.01)

    def test_uniform_spacing(self):
        """Output should be uniformly spaced."""
        times = np.array([0, 1.0, 2.0, 3.0])
        rri = np.array([1000, 950, 1050, 980])
        rri_interp, times_interp = interpolate_rri(rri, times, step_s=0.1)
        diffs = np.diff(times_interp)
        np.testing.assert_allclose(diffs, 0.1, atol=1e-10)
