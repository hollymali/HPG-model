"""Plot nighttime E2 pulse amplitude over the Figure 2 steepness grid."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

from steepness_2d_heatmaps import AMPLITUDE, AMPLITUDE_DEFAULT, LH_STEEPNESS, LH_DEFAULT
from MODEL import START_CLOCK_H, STEP_H, WARMUP_H, simulate_endocrine


HERE = Path(__file__).resolve().parent
PNG = HERE / "figure4.png"
FFT_DAYS = 42
START_42_DAY_H = WARMUP_H
STOP_H = START_42_DAY_H + FFT_DAYS * 24


def nighttime_excursions(e2, start_h):
    """Measure peaks at 23:00–07:00 using the smaller of the two trough drops."""
    middle = e2[1:-1]
    peaks = np.flatnonzero((middle > e2[:-2]) & (middle >= e2[2:])) + 1
    troughs = np.flatnonzero((middle < e2[:-2]) & (middle <= e2[2:])) + 1

    # Use integer steps so 23:00 is included and 07:00 is excluded exactly.
    peak_steps = peaks + round(start_h / STEP_H)
    steps_per_day = round(24 / STEP_H)
    clock_steps = (peak_steps + round(START_CLOCK_H / STEP_H)) % steps_per_day
    asleep = (clock_steps >= round(23 / STEP_H)) | (clock_steps < round(7 / STEP_H))
    in_window = (round(START_42_DAY_H / STEP_H) <= peak_steps) & (
        peak_steps < round(STOP_H / STEP_H)
    )
    peaks = peaks[asleep & in_window]

    # A neighboring trough can fall outside the sleep window.
    right = np.searchsorted(troughs, peaks)
    paired = (right > 0) & (right < len(troughs))
    peaks, right = peaks[paired], right[paired]
    return e2[peaks] - np.maximum(e2[troughs[right - 1]], e2[troughs[right]])


def simulate_grid(pairs):
    """Keep the full 1.5 s trace in small batches to resolve peaks and troughs."""
    medians = np.empty(len(pairs))
    first = round(WARMUP_H / STEP_H)
    # Normal mode retains all variables, so use smaller batches to bound memory.
    for start in range(0, len(pairs), 4):
        stop = min(start + 4, len(pairs))
        lh, amplitude = pairs[start:stop].T
        model = simulate_endocrine(
            output_h=STOP_H + 24 - WARMUP_H,
            inhibition_exponent=lh, n_amplitude_exponent=amplitude,
        )
        traces = model["E2"][..., first:]
        del model
        for row, trace in enumerate(traces, start=start):
            excursions = nighttime_excursions(trace, WARMUP_H)
            medians[row] = np.median(excursions)
        print(f"Completed {stop}/{len(pairs)} parameter combinations", flush=True)
    return medians


def fold_change(pairs, medians):
    baseline_index, = np.flatnonzero((pairs[:, 0] == LH_DEFAULT) & (pairs[:, 1] == AMPLITUDE_DEFAULT))
    baseline = medians[baseline_index]
    if not np.isfinite(medians).all() or (medians <= 0).any():
        raise ValueError("Expected finite, positive median amplitudes")
    return (medians / baseline).reshape(len(AMPLITUDE), len(LH_STEEPNESS)), baseline


def save_figure(fold):
    fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")
    mesh = ax.pcolormesh(LH_STEEPNESS, AMPLITUDE, fold, shading="nearest",
                         cmap="RdBu_r", norm=LogNorm(vmin=1 / 8, vmax=8))
    ax.plot(LH_DEFAULT, AMPLITUDE_DEFAULT, "o", color="white", markeredgecolor="black",
            clip_on=False, label=f"Baseline: LH {LH_DEFAULT:g}, amplitude {AMPLITUDE_DEFAULT:g}")
    ax.set(xlim=(0.975, 5.025), ylim=(0.975, 5.025))
    ax.spines[["top", "right"]].set_visible(False)
    ticks = 2.0**np.arange(-3, 4)
    bar = fig.colorbar(mesh, ax=ax, extend="both", ticks=ticks)
    bar.ax.set_yticklabels([f"{tick:g}\u00d7" for tick in ticks])
    bar.minorticks_off()
    bar.set_label("Median nighttime E2 amplitude / baseline")
    fig.supxlabel("LH desensitization steepness")
    fig.supylabel("N\u2192GnRH pulse amplitude steepness")
    fig.suptitle("Median Nighttime E2 Pulse Amplitude", fontweight="bold")
    ax.legend(frameon=False, loc="upper left")
    fig.savefig(PNG, dpi=300)
    plt.close(fig)


def main():
    lh, amplitude = np.meshgrid(LH_STEEPNESS, AMPLITUDE)
    pairs = np.column_stack((lh.ravel(), amplitude.ravel()))
    medians = simulate_grid(pairs)
    fold, baseline = fold_change(pairs, medians)
    save_figure(fold)
    print(f"Baseline median E2 pulse amplitude: {baseline:.3f} pg/mL")


if __name__ == "__main__":
    main()
