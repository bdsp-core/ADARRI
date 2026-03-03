#!/usr/bin/env python3
"""HRV spectrogram before and after ADARRI artifact reduction.

Two-panel figure showing the effect of artifact removal on HRV analysis,
matching the style of Nagaraj et al. (2016) Figure 2:
  (a) Raw RRI time series + spectrogram (before artifact reduction)
  (b) Cleaned RRI + spectrogram (after ADARRI artifact reduction)

The cleaning uses the absolute difference of adjacent RRI (adRRI) to identify
artifacts: points where |diff(RRI)| exceeds the 95th percentile are removed
and the gaps are filled by linear interpolation.

Uses continuous IBI data from one ICU patient (~20 days).

Port of matlab/FigureForPaper_Example/a_Figure_HRV_spectrogram_v2.m.

Usage:
    python scripts/figure_hrv_spectrogram.py --data data/ibi_for_Dennis.mat --output outputs/
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def load_ibi(filepath):
    """Load IBI data from .mat file.

    Returns:
        t_days: time in days
        rri_sec: RR intervals in seconds
    """
    mat = loadmat(filepath)
    ib = mat["ib_final"]
    t_hours = ib[:, 0]
    rri_sec = ib[:, 1]
    t_days = t_hours / 24.0
    return t_days, rri_sec


def clean_rri(t_days, rri_sec, quantile_threshold=0.95):
    """Remove artifact beats and interpolate across gaps.

    Mirrors the MATLAB approach (a_Figure_HRV_spectrogram_v2.m):
    remove points where |diff(RRI)| exceeds the specified quantile
    of abs(diff(RRI)), then linearly interpolate.
    """
    yd = np.concatenate([[0], np.abs(np.diff(rri_sec))])
    th = np.quantile(yd, quantile_threshold)

    keep = yd < th
    t_clean = t_days[keep]
    rri_clean = rri_sec[keep]

    # Interpolate back onto the original time grid
    rri_interp = np.interp(t_days, t_clean, rri_clean)
    return rri_interp


def compute_spectrogram(t_days, rri_sec, window_sec=120, step_samples=50,
                        max_freq=0.5):
    """Compute spectrogram of RRI time series.

    Faithfully ports the MATLAB code (a_Figure_HRV_spectrogram_v2.m):
    resample onto uniform grid (5x oversampling), sliding-window FFT
    with detrending, normalize by total sample count (matching MATLAB
    fft/Nt convention), then 2D Gaussian smoothing of the spectrogram.

    Args:
        t_days: time in days
        rri_sec: RRI in seconds
        window_sec: FFT window length in seconds (default 120 = 2 min)
        step_samples: step between windows in resampled samples
        max_freq: maximum frequency to include (Hz)

    Returns:
        spec_db: spectrogram in dB (freq x time)
        t_spec_days: time axis in days
        f_spec: frequency axis in Hz
    """
    # Resample onto uniform grid (5x oversampling), matching MATLAB:
    # Nt=length(t)*5; T=max(t)-min(t); dt=T/Nt; ti=(0:Nt-1)*dt;
    n_resamp = len(t_days) * 5
    T = t_days[-1] - t_days[0]  # in days
    dt_days = T / n_resamp
    t_uniform = np.arange(n_resamp) * dt_days + t_days[0]
    rri_uniform = np.interp(t_uniform, t_days, rri_sec)

    # Remove NaNs (matching MATLAB: ind=find(isnan(y)); y(ind)=[]; t(ind)=[];)
    valid = ~np.isnan(rri_uniform)
    t_uniform = t_uniform[valid]
    rri_uniform = rri_uniform[valid]

    # Convert to seconds for spectral analysis (matching MATLAB: t=t*24*60*60)
    t_sec = t_uniform * 24 * 60 * 60
    dt = t_sec[1] - t_sec[0]
    fs = 1.0 / dt
    nt = int(round(window_sec / dt))
    N = len(t_sec)

    nfft = int(2 ** np.ceil(np.log2(nt)))

    # Frequency axis: positive frequencies only
    # Matching MATLAB: f = Fs/2*linspace(0,1,NFFT/2+1)
    f = fs / 2 * np.linspace(0, 1, nfft // 2 + 1)
    freq_mask = f < max_freq
    f_spec = f[freq_mask]

    # Sliding window FFT
    spec_list = []
    t_spec = []

    n_windows = (N - nt) // step_samples
    report_interval = max(1, n_windows // 20)

    for idx, i in enumerate(range(0, N - nt, step_samples)):
        segment = rri_uniform[i:i + nt]
        # Detrend then demean (matching MATLAB: yt=detrend(yt); yt=yt-mean(yt))
        x = np.arange(len(segment), dtype=float)
        coeffs = np.polyfit(x, segment, 1)
        segment = segment - np.polyval(coeffs, x)
        segment = segment - np.mean(segment)

        # FFT normalized by total N (matching MATLAB: Y = fft(yt,NFFT)/Nt)
        Y = np.fft.fft(segment, nfft) / N
        # Power spectrum (matching MATLAB: S=Y.*conj(Y))
        S = np.real(Y * np.conj(Y))

        # Keep positive frequencies up to max_freq
        # Matching MATLAB: S=S(1:NFFT/2+1); ind=find(f<0.5); I(:,ct)=S(ind)
        S_pos = S[:nfft // 2 + 1]
        S_pos = S_pos[freq_mask]

        spec_list.append(S_pos)
        t_spec.append(t_uniform[i])  # keep in days

        if idx % report_interval == 0:
            print(f"  {idx}/{n_windows} windows ({100*idx/n_windows:.0f}%)")

    spec = np.array(spec_list).T  # (freq x time)
    t_spec_days = np.array(t_spec)

    # Convert to dB (matching MATLAB: pow2db(Iblur))
    spec_db = 10 * np.log10(np.maximum(spec, 1e-30))

    # 2D Gaussian smoothing (matching MATLAB: imgaussfilt(I,[2 1]))
    spec_db = gaussian_filter(spec_db, sigma=[2, 1])

    return spec_db, t_spec_days, f_spec


def make_figure(data_path, output_dir):
    """Generate the before/after HRV spectrogram figure."""
    os.makedirs(output_dir, exist_ok=True)

    print("Loading IBI data...")
    t_days, rri_raw = load_ibi(data_path)
    print(f"  {len(t_days)} samples, {t_days[-1]:.1f} days")

    print("Cleaning RRI (removing artifacts via adRRI threshold)...")
    rri_clean = clean_rri(t_days, rri_raw)

    print("Computing spectrogram of RAW RRI...")
    spec_raw, t_days_spec, f_hz = compute_spectrogram(t_days, rri_raw)

    print("Computing spectrogram of CLEANED RRI...")
    spec_clean, _, _ = compute_spectrogram(t_days, rri_clean)

    # Color scale (matching MATLAB: [-170 -120])
    vmin = -170
    vmax = -120

    # Layout: 4 subplots (RRI + spectrogram for each of 2 panels)
    # Use GridSpec for tighter control of spacing between groups
    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(4, 1, height_ratios=[1, 1.3, 1, 1.3],
                          hspace=0.08, left=0.10, right=0.95,
                          top=0.97, bottom=0.06)
    # Add extra gap between panels (a) and (b)
    gs.update(hspace=0.08)
    axes = [fig.add_subplot(gs[i]) for i in range(4)]

    # Add vertical gap between the two groups
    pos1 = axes[1].get_position()
    pos2 = axes[2].get_position()
    gap = 0.04
    for i in [2, 3]:
        p = axes[i].get_position()
        axes[i].set_position([p.x0, p.y0 - gap, p.width, p.height])

    t_max = t_days[-1]

    # === Panel (a): Raw ===
    # RRI time series
    ax = axes[0]
    ax.plot(t_days, rri_raw, "b-", linewidth=0.2)
    ax.set_ylabel("RRI (sec)", fontsize=10)
    ax.set_ylim([0, 1.5])
    ax.set_xlim([0, t_max])
    ax.tick_params(labelbottom=False)

    # Spectrogram
    ax = axes[1]
    ax.pcolormesh(t_days_spec, f_hz, spec_raw, vmin=vmin, vmax=vmax,
                  cmap="jet", shading="auto", rasterized=True)
    ax.set_ylabel("Frequency (Hz)", fontsize=10)
    ax.set_xlim([0, t_max])
    ax.set_xlabel("Time (days)", fontsize=10)
    ax.text(0.5, -0.22, "(a)", transform=ax.transAxes,
            fontsize=12, fontweight="bold", ha="center")

    # === Panel (b): Cleaned ===
    # RRI time series
    ax = axes[2]
    ax.plot(t_days, rri_clean, "b-", linewidth=0.2)
    ax.set_ylabel("RRI (sec)", fontsize=10)
    ax.set_ylim([0, 1.5])
    ax.set_xlim([0, t_max])
    ax.tick_params(labelbottom=False)

    # Spectrogram
    ax = axes[3]
    im = ax.pcolormesh(t_days_spec, f_hz, spec_clean, vmin=vmin, vmax=vmax,
                       cmap="jet", shading="auto", rasterized=True)
    ax.set_ylabel("Frequency (Hz)", fontsize=10)
    ax.set_xlim([0, t_max])
    ax.set_xlabel("Time (days)", fontsize=10)
    ax.text(0.5, -0.22, "(b)", transform=ax.transAxes,
            fontsize=12, fontweight="bold", ha="center")

    outpath = os.path.join(output_dir, "figure_hrv_spectrogram.png")
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    print(f"Saved to {outpath}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/ibi_for_Dennis.mat")
    parser.add_argument("--output", default="outputs/")
    args = parser.parse_args()
    make_figure(args.data, args.output)
