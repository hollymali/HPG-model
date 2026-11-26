"""FFT power by period for regularly sampled E2 traces."""

import numpy as np


SAMPLE_MINUTES = 7
PERIODS = np.array([1, 2, 4, 8, 12, 24])


def fft_power(samples, sample_minutes=SAMPLE_MINUTES):
    """Return shared periods (hours) and per-trace powers (|rFFT|^2).

    The last axis of samples is time. The last axis of power indexes FFT bins.
    The 1D periods array is shared by all traces. The zero-frequency bin is omitted.
    """
    samples = np.asarray(samples)
    count = samples.shape[-1]
    frequencies = np.fft.rfftfreq(count, d=sample_minutes / 60)
    periods = 1 / frequencies[1:]
    coefficients = np.fft.rfft(samples, axis=-1)
    power = np.abs(coefficients[..., 1:]) ** 2
    return periods, power


def band_power(periods, power, target_periods=PERIODS):
    """Sum power within ±5% of each target period, one sum per trace and target."""
    bands = []
    for target in target_periods:
        in_band = (0.95 * target <= periods) & (periods <= 1.05 * target)
        total = power[..., in_band].sum(axis=-1)
        bands.append(total)
    return np.stack(bands, axis=-1)
