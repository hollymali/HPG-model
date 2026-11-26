"""Sweep SCN minima from 25 to 45 in 0.1 steps and plot E2 band power.

Use 618 seven-minute samples starting at 08:00 after a 96-hour warmup.
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
SCN_DEFAULT = REFERENCE["scn_min"]
MINIMA = np.linspace(25, 45, 201)
COLOR_TICKS = (0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3)


def main():
    samples = simulate_endocrine(for_fft=True, output_h=72, scn_min=MINIMA)
    periods, bin_power = fft_power(samples)
    power = band_power(periods, bin_power)
    fold = power / power[MINIMA == SCN_DEFAULT]

    ticks = np.arange(25, 46, 2)
    rows = np.arange(len(PERIODS))
    low, high = COLOR_TICKS[0], COLOR_TICKS[-1]
    fig, ax = plt.subplots(figsize=(10, 4.5), layout="constrained")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("B. SCN Effect", loc="left", fontsize=18, fontweight="bold", pad=10)
    mesh = ax.pcolormesh(
        MINIMA, rows, fold.T, shading="nearest", cmap="RdBu_r",
        norm=LogNorm(vmin=low, vmax=high),
    )
    ax.set_yticks(rows, [f"{period:g}" for period in PERIODS])
    ax.set_xticks(ticks, [f"{tick:g}" for tick in ticks])
    ax.set(ylim=(len(PERIODS) - 0.5, -0.5),
           xlabel="SCN minimum", ylabel="E2 rhythm period (h)")
    ax.axvline(SCN_DEFAULT, color="#333333", linestyle="--", linewidth=1,
               path_effects=[withStroke(linewidth=2, foreground="white")])
    ax.text(SCN_DEFAULT, 1.02, f"Baseline value = {SCN_DEFAULT:g}",
            transform=ax.get_xaxis_transform(), ha="center", va="bottom")

    bar = fig.colorbar(mesh, ax=ax, extend="both", pad=0.025, aspect=19, ticks=COLOR_TICKS)
    bar.ax.set_yticklabels([f"{tick:g}\u00d7" for tick in COLOR_TICKS])
    bar.minorticks_off()
    bar.set_label("Band power / baseline band power")
    fig.savefig(HERE / "scn_minimum_rhythm_heatmap.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
