""" Tools for visualizing data and the results of modeling fitting. """

from typing import Callable, Optional, Sequence

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import matplotlib as mpl

from janelia_core.math.basic_functions import find_binary_runs

# Define aliases
OptionalAxes = Optional[plt.Axes]
OptionalCallable = Optional[Callable]
OptionalMultipleAxes = Optional[Sequence[plt.Axes]]

def import_style():
    mpl.style.use(Path(__file__).parent / "rc_params.mplstyle")

import_style()


def plot_segmented_signal(tm_pts: np.ndarray, sig: np.ndarray, ax: plt.Axes = None, delta: float = .6,
                          remove_tm_btw_chunks: bool = True, tm_padding: float = 1.0, color='k',
                          linewidth=1.0):
    """ Plots a signal that exists at different chunks in time.

    Each chunk will be plotted as a seperate trace, and the user can chose to remove time between chunks.

    Args:
        tm_pts: The array of time points the signal is sampled at

        sig: The array of signal values

        ax: Axes to plot into.  If None, a figure with axes will be crated.

        delta: The threshold to use when determining if two sequential time points belong to the same chunk or not
    """

    if ax is None:
        f = plt.figure()
        ax = plt.subplot(1,1,1)

    # Make sure everything is sorted
    sort_order = np.argsort(tm_pts)
    tm_pts = tm_pts[sort_order]
    sig = sig[sort_order]

    # Find the chunks
    tm_diff = np.diff(tm_pts)
    small_diffs = tm_diff < delta
    runs = find_binary_runs(small_diffs)
    for r_i, run in enumerate(runs):
        runs[r_i] = slice(run.start, run.stop+1)

    # Remove time between chunks if we are suppose to
    if remove_tm_btw_chunks:
        cur_adj = 0
        for run in runs:
            cur_tm_span = tm_pts[run][-1] - tm_pts[run][0]
            tm_pts[run] = (tm_pts[run] - tm_pts[run][0]) + cur_adj
            cur_adj += cur_tm_span + tm_padding

    for run in runs:
        ax.plot(tm_pts[run], sig[run], color=color, linewidth=linewidth)

    return ax


def make_blue_red_c_map(n: int = 256, inc_transp: bool = False,
                        gentle: bool = False) -> mcolors.LinearSegmentedColormap:
    """Generates a color map that linearly goes from blue at 0, to black at 0.5, to red at 1.

    Args:
        n: The number of values in the color map.
        inc_transp: True if values in the middle of the map (black) should also be transparent.
        gentle: If True, adds gentle fade in/out near the center.

    Returns:
        cmap: The generated color map.
    """
    if inc_transp:
        middle_alpha = 0.0
    else:
        middle_alpha = 1.0

    # RGB: Blue = (0, 0, 1), Red = (1, 0, 0), Black = (0, 0, 0)
    if not gentle:
        return mcolors.LinearSegmentedColormap.from_list(
            name='blue_to_red',
            colors=[
                (0.0, [0.0, 0.0, 1.0, 1.0]),              # Blue
                (0.5, [0.0, 0.0, 0.0, middle_alpha]),      # Black (transparent if specified)
                (1.0, [1.0, 0.0, 0.0, 1.0])               # Red
            ],
            N=n
        )
    else:
        return mcolors.LinearSegmentedColormap.from_list(
            name='blue_to_red',
            colors=[
                (0.0,  [0.0, 0.0, 1.0, 1.0]),               # Blue
                (0.47,  [0.0, 0.0, 1.0, 0.10]),               # Fade-out blue
                (0.5,  [0.0, 0.0, 0.0, middle_alpha]),      # Black
                (0.53,  [1.0, 0.0, 0.0, 0.10]),               # Fade-in red
                (1.0,  [1.0, 0.0, 0.0, 1.0])                # Red
            ],
            N=n
        )
