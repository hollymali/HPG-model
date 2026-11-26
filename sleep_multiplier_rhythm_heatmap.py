"""Sweep sleep multipliers from 1 to 2 in 0.005 steps and plot E2 band power.

Use 42 days of seven-minute samples starting at 08:00 after a 96-hour warmup.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patheffects import withStroke
import numpy as np

from MODEL import REFERENCE, simulate_endocrine
from rhythm_fft import PERIODS, band_power, fft_power


HERE = Path(__file__).resolve().parent
SLEEP_DEFAULT = REFERENCE["sleep_multiplier"]
MULTIPLIERS = np.linspace(1, 2, 201)
COLOR_TICKS = (0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3)


def main():
    samples = simulate_endocrine(for_fft=True, output_h=42 * 24, sleep_multiplier=MULTIPLIERS)
    periods, bin_power = fft_power(samples)
    power = band_power(periods, bin_power)
    fold = power / power[MULTIPLIERS == SLEEP_DEFAULT]

    ticks = np.linspace(1, 2, 11)
    rows = np.arange(len(PERIODS))
    low, high = COLOR_TICKS[0], COLOR_TICKS[-1]
    fig, ax = plt.subplots(figsize=(10, 4.5), layout="constrained")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("C. Sleep Effect", loc="left", fontsize=18, fontweight="bold", pad=10)
    mesh = ax.pcolormesh(
        MULTIPLIERS, rows, fold.T, shading="nearest", cmap="RdBu_r",
        norm=LogNorm(vmin=low, vmax=high),
    )
    ax.set_yticks(rows, [f"{period:g}" for period in PERIODS])
    ax.set_xticks(ticks, [f"{tick:.1f}\u00d7" for tick in ticks])
    ax.set(ylim=(len(PERIODS) - 0.5, -0.5),
           xlabel="Nighttime sleep multiplier", ylabel="E2 rhythm period (h)")
    ax.axvline(SLEEP_DEFAULT, color="#333333", linestyle="--", linewidth=1,
               path_effects=[withStroke(linewidth=2, foreground="white")])
    ax.text(SLEEP_DEFAULT, 1.02, f"Baseline value = {SLEEP_DEFAULT:g}\u00d7",
            transform=ax.get_xaxis_transform(), ha="center", va="bottom")

    bar = fig.colorbar(mesh, ax=ax, extend="both", pad=0.025, aspect=19, ticks=COLOR_TICKS)
    bar.ax.set_yticklabels([f"{tick:g}\u00d7" for tick in COLOR_TICKS])
    bar.minorticks_off()
    bar.set_label("Band power / baseline band power")
    fig.savefig(HERE / "sleep_multiplier_rhythm_heatmap.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
