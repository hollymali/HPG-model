"""Plot three-day E2 and LH traces and E2 FFT power for Meeting 15 and Licinio.

Run from the project folder: python meeting15/figure1.py
"""

from pathlib import Path

import matplotlib
import numpy as np

from digitize_licinio import digitize_licinio, digitize_licinio_lh
from MODEL import STEP_H, WARMUP_H, simulate_endocrine
from rhythm_fft import PERIODS, SAMPLE_MINUTES, band_power, fft_power

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent


def main():
    # 1. Read the two digitized traces. Each has time and concentration columns.
    licinio_hours, licinio_e2 = digitize_licinio().T
    licinio_lh_hours, licinio_lh = digitize_licinio_lh().T

    # 2. Run three days for the graph. The 96-hour warmup ends at 08:00.
    days = 3
    output_hours = days * 24
    model = simulate_endocrine(output_h=output_hours)
    start = round(WARMUP_H / STEP_H)
    model_e2 = model["E2"][start:]
    model_lh = model["LH"][start:]
    model_hours = model["time_h"][start:] - WARMUP_H

    # 3. Run three days for the FFT, keeping E2 at seven-minute intervals.
    model_samples = simulate_endocrine(for_fft=True, output_h=output_hours)
    fft_times = np.arange(len(model_samples)) * (SAMPLE_MINUTES / 60)
    # Licinio supplies one day; periodic interpolation repeats it every 24 h.
    licinio_samples = np.interp(fft_times, licinio_hours, licinio_e2, period=24)
    # Analyze both traces together, then sum power within ±5% of each target period.
    periods, bin_power = fft_power([model_samples, licinio_samples])
    model_power, licinio_power = band_power(periods, bin_power)

    # 4. Plot three days of E2 and LH.
    e2_color, lh_color, licinio_color = "#2166a0", "#b62932", "#606060"
    fig, (e2_ax, lh_ax, power_ax) = plt.subplots(3, 1, figsize=(10.8, 8), layout="constrained")
    e2_ax.sharex(lh_ax)
    e2_ax.plot(model_hours, model_e2, color=e2_color, label="Baseline model", lw=1.5)
    lh_ax.plot(model_hours, model_lh, color=lh_color, label="Baseline model", lw=1.5)

    licinio_label = "Licinio et al. (one day repeated)"
    for day in range(days):
        day_start = day * 24
        label = licinio_label if day == 0 else None
        e2_ax.plot(licinio_hours + day_start, licinio_e2, color=licinio_color, label=label, lw=1.3)
        lh_ax.plot(licinio_lh_hours + day_start, licinio_lh, color=licinio_color, label=label, lw=1.3)
        for ax in (e2_ax, lh_ax):
            ax.axvspan(day_start + 15, day_start + 23, color="0.92", zorder=0)  # 23:00 to 07:00.

    # Leave space above each trace so its legend stays clear of the peaks.
    e2_ax.set(ylabel="E2 (pg/mL)", ylim=(0, 1.2 * max(model_e2.max(), licinio_e2.max())))
    e2_ax.tick_params(labelbottom=False)
    e2_ax.legend(frameon=False, ncol=2, loc="upper right")

    lh_ax.set(xlim=(0, output_hours), xlabel="Clock time", ylabel="LH (IU/L)",
              ylim=(0, 1.2 * max(model_lh.max(), licinio_lh.max())))
    lh_ax.legend(frameon=False, ncol=2, loc="upper left")
    ticks = np.arange(0, output_hours + 1, 8)
    lh_ax.set_xticks(ticks, [f"{(8 + hour) % 24:02d}:00" for hour in ticks])

    # 5. Plot the six E2 rhythm-power bands side by side on a logarithmic axis.
    positions = np.arange(len(PERIODS))
    width = 0.36
    offset = 0.22
    model_bars = power_ax.bar(positions - offset, model_power, width, color=e2_color, label="Baseline model")
    licinio_bars = power_ax.bar(positions + offset, licinio_power, width, color=licinio_color,
                               label=licinio_label)
    power_ax.set_xticks(positions, PERIODS)
    power_ax.set(xlabel="Target period (h) with ±5% bands",
                 ylabel=r"FFT power ($|\mathrm{FFT}|^2$)", yscale="log")
    power_ax.margins(y=0.15)
    for bars in (model_bars, licinio_bars):
        labels = [rf"$10^{{{np.log10(power):.2f}}}$" for power in bars.datavalues]
        power_ax.bar_label(bars, labels=labels, padding=3, fontsize=8)
    power_ax.legend(frameon=False, loc="upper left", ncol=2)

    for ax, title in zip((e2_ax, lh_ax, power_ax),
                         ("A. Estradiol", "B. Luteinizing Hormone", "C. Estradiol Rhythm Power (3-Day FFT)")):
        ax.set_title(title, loc="left", fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)

    # 6. Save beside this script, regardless of the working directory.
    fig.savefig(HERE / "figure1.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
