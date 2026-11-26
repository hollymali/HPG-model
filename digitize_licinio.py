"""Digitize the Subject 2 E2 and LH panels from Licinio et al. (1998)."""

from pathlib import Path

import numpy as np
from PIL import Image

from rhythm_fft import SAMPLE_MINUTES


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "Licinio_E2_LH.png"


def _digitize_panel(*, x_crop, y_crop, x_axis, y_axis, axis_max):
    x_start, x_stop = x_crop
    y_start, y_stop = y_crop
    with Image.open(SOURCE) as image:
        gray = np.asarray(image.crop((x_start, y_start, x_stop, y_stop)).convert("L"))
    # The median dark pixel in each column estimates the center of the line.
    dark = gray < gray.max() / 2
    trace_y = np.array([np.median(np.flatnonzero(column)) for column in dark.T]) + y_start

    hours = np.arange(0, 24 * 60, SAMPLE_MINUTES) / 60
    x_first, x_last = x_axis
    sample_x = x_first + (x_last - x_first) * hours / 24
    # The 08:00 sample uses the first clear column, just right of the axis.
    sample_y = np.interp(sample_x, np.arange(x_start, x_stop), trace_y)
    y_zero, y_top = y_axis
    concentration = axis_max * (y_zero - sample_y) / (y_zero - y_top)
    return np.column_stack((hours, concentration))


def digitize_licinio():
    """Return two columns: hours since 08:00 and approximate E2 in pg/mL."""
    return _digitize_panel(x_crop=(116, 353), y_crop=(592, 743),
                           x_axis=(115, 351), y_axis=(745.5, 579), axis_max=60)


def digitize_licinio_lh():
    """Return two columns: hours since 08:00 and approximate LH in IU/L."""
    return _digitize_panel(x_crop=(115, 351), y_crop=(364, 512),
                           x_axis=(114, 350), y_axis=(513, 353), axis_max=7)


if __name__ == "__main__":
    np.save(HERE / "licinio_e2.npy", digitize_licinio())
    np.save(HERE / "licinio_lh.npy", digitize_licinio_lh())
