"""Simulate the joint steepness sweep and draw relative E2 power."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

from MODEL import REFERENCE, simulate_endocrine
from rhythm_fft import PERIODS, band_power, fft_power


HERE = Path(__file__).resolve().parent
NAME = "lh_amplitude_joint_heatmaps"
FFT_DAYS = 42
LH_DEFAULT = REFERENCE["inhibition_exponent"]
AMPLITUDE_DEFAULT = REFERENCE["n_amplitude_exponent"]
LH_STEEPNESS = np.round(np.arange(1, 5.01, .01), 2)
AMPLITUDE = LH_STEEPNESS.copy()


def simulate_grid():
    """Sweep in batches of amplitude rows to keep memory use bounded."""
    power = np.empty((len(AMPLITUDE), len(LH_STEEPNESS), len(PERIODS)))
    for start in range(0, len(AMPLITUDE), 8):
        stop = min(start + 8, len(AMPLITUDE))
        samples = simulate_endocrine(
            for_fft=True, output_h=FFT_DAYS * 24, inhibition_exponent=LH_STEEPNESS,
            n_amplitude_exponent=AMPLITUDE[start:stop, None],
        )
        periods, bin_power = fft_power(samples)
        power[start:stop, :, :] = band_power(periods, bin_power)
        print(f"Completed {stop}/{len(AMPLITUDE)} amplitude rows", flush=True)
    return power


def save_figures(fold):
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True, sharey=True,
                             layout="constrained")
    for band, (ax, period) in enumerate(zip(axes.flat, PERIODS)):
        mesh = ax.pcolormesh(LH_STEEPNESS, AMPLITUDE, fold[:, :, band], shading="nearest",
                             cmap="RdBu_r", norm=LogNorm(vmin=1 / 8, vmax=8))
        ax.plot(LH_DEFAULT, AMPLITUDE_DEFAULT, "o", color="white", markeredgecolor="black",
                clip_on=False, label=f"Baseline: LH {LH_DEFAULT:g}, amplitude {AMPLITUDE_DEFAULT:g}")
        ax.set(xlim=(0.975, 5.025), ylim=(0.975, 5.025))
        ax.set_title(f"{period:g} h", loc="left", fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
    ticks = 2.0**np.arange(-3, 4)
    bar = fig.colorbar(mesh, ax=axes, extend="both", ticks=ticks)
    bar.ax.set_yticklabels([f"{tick:g}\u00d7" for tick in ticks])
    bar.minorticks_off()
    bar.set_label("Band power / baseline band power")
    fig.supxlabel("LH desensitization steepness")
    fig.supylabel("N\u2192GnRH pulse amplitude steepness")
    fig.suptitle("A. Pulse Amplitude + LH Desensitization Steepness Effect",
                 fontweight="bold")
    axes[0, 0].legend(frameon=False, loc="upper left")
    fig.savefig(HERE / f"{NAME}_relative.png", dpi=300)
    plt.close(fig)


def main():
    power = simulate_grid()
    baseline_power = power[AMPLITUDE == AMPLITUDE_DEFAULT, LH_STEEPNESS == LH_DEFAULT, :]
    fold = power / baseline_power

    save_figures(fold)


if __name__ == "__main__":
    main()
