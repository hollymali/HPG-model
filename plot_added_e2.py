"""Plot three-day E2 traces and FFT power for baseline and added E2.

Run from the project folder: python meeting15/plot_added_e2.py
"""

from pathlib import Path

import matplotlib
import numpy as np

from MODEL import STEP_H, WARMUP_H, simulate_endocrine
from rhythm_fft import PERIODS, band_power, fft_power

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent


def create_figure():
    # 1. Run three days for the graph. The 96-hour warmup ends at 08:00.
    days = 3
    output_hours = days * 24
    added_e2 = np.array([0, 30, 100])
    model = simulate_endocrine(output_h=output_hours, added_e2=added_e2)
    start = round(WARMUP_H / STEP_H)
    # MODEL returns ovarian E2. Add the fixed dose to plot total E2.
    model_e2 = model["E2"][:, start:] + added_e2[:, None]
    model_hours = model["time_h"][start:] - WARMUP_H

    # 2. Run three days for the FFT, keeping E2 at seven-minute intervals.
    model_samples = simulate_endocrine(for_fft=True, output_h=output_hours, added_e2=added_e2)
    model_samples = model_samples + added_e2[:, None]
    # Analyze all three traces together, then sum power within ±5% of each target period.
    periods, bin_power = fft_power(model_samples)
    model_power = band_power(periods, bin_power)

    # 3. Plot three days of E2, comparing baseline with one dose per row.
    e2_color, added_color = "#2166a0", "#d88732"
    fig, axes = plt.subplots(2, 2, figsize=(14, 7.2), sharex="col", layout="constrained")
    e2_axes = axes[:, 0]
    power_axes = axes[:, 1]
    power_axes[0].sharey(power_axes[1])
    added_labels = [f"+{amount} pg/mL added E2" for amount in added_e2[1:]]

    baseline_e2 = model_e2[0]
    for e2_ax, added_trace, added_label in zip(e2_axes, model_e2[1:], added_labels):
        e2_ax.plot(model_hours, baseline_e2, color=e2_color, label="Baseline model", lw=1.5)
        e2_ax.plot(model_hours, added_trace, color=added_color, label=added_label, lw=1.3)

        for day in range(days):
            day_start = day * 24
            e2_ax.axvspan(day_start + 15, day_start + 23, color="0.92", zorder=0)  # 23:00 to 07:00.

        # Leave space above each trace so its legend stays clear of the peaks.
        e2_ax.set(xlim=(0, output_hours), ylabel="E2 (pg/mL)",
                  ylim=(0, 1.2 * max(baseline_e2.max(), added_trace.max())))
        e2_ax.legend(frameon=False, ncol=2, loc="upper right")

    ticks = np.arange(0, output_hours + 1, 8)
    e2_axes[1].set_xlabel("Clock time")
    e2_axes[1].set_xticks(ticks, [f"{(8 + hour) % 24:02d}:00" for hour in ticks])

    # 4. Plot the six E2 rhythm-power bands side by side on a logarithmic axis.
    positions = np.arange(len(PERIODS))
    width = 0.36
    offset = 0.22
    baseline_power = model_power[0]
    for power_ax, added_power, added_label in zip(power_axes, model_power[1:], added_labels):
        baseline_bars = power_ax.bar(positions - offset, baseline_power, width, color=e2_color,
                                     label="Baseline model")
        added_bars = power_ax.bar(positions + offset, added_power, width, color=added_color,
                                  label=added_label)
        power_ax.set_xticks(positions, PERIODS)
        power_ax.set(ylabel=r"FFT power ($|\mathrm{FFT}|^2$)", yscale="log")
        power_ax.margins(y=0.15)
        for bars in (baseline_bars, added_bars):
            labels = [rf"$10^{{{np.log10(power):.2f}}}$" for power in bars.datavalues]
            power_ax.bar_label(bars, labels=labels, padding=3, fontsize=8)
        power_ax.legend(frameon=False, loc="upper left", ncol=2)

    power_axes[1].set_xlabel("Target period (h) with ±5% bands")

    for ax, title in zip(axes.flat, ("A. Added E2 (+30 pg/mL)", "Estradiol Rhythm Power (3-Day FFT)",
                                     "B. Added E2 (+100 pg/mL)", "Estradiol Rhythm Power (3-Day FFT)")):
        ax.set_title(title, loc="left", fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)

    # 5. Save beside this script, regardless of the working directory.
    out = HERE / "added_e2_30_100.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    for amount, samples, power in zip(added_e2[1:], model_samples[1:], model_power[1:]):
        print(f"Added E2 {amount:3d} pg/mL; mean total: {samples.mean():.2f} pg/mL; "
              f"log10 FFT band powers: {np.round(np.log10(power), 2)}")
    print(f"Baseline mean {model_samples[0].mean():.2f} pg/mL")
    print(out)
    return out


if __name__ == "__main__":
    create_figure()
