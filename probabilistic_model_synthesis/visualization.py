""" Tools for visualizing data and the results of modeling fitting. """

from typing import Callable, Optional, Sequence, Union

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import torch

from janelia_core.math.basic_functions import bound
from janelia_core.visualization.image_generation import generate_2d_fcn_image
from janelia_core.math.basic_functions import find_binary_runs
from janelia_core.math.basic_functions import list_grid_pts
from janelia_core.ml.torch_distributions import CondVAEDistribution
from janelia_core.ml.utils import torch_mod_to_fcn

# Define aliases
OptionalAxes = Optional[plt.Axes]
OptionalCallable = Optional[Callable]
OptionalMultipleAxes = Optional[Sequence[plt.Axes]]


def assign_colors_to_pts(pts: np.ndarray, lims: np.ndarray) -> np.ndarray:
    """ Assigns colors to points based on their location.

    This function can work with points in 2-d or 3-d.  The red, blue and green channels of a points color are functions
    of the first, second and third (if present) coordinates of that point. For each dimension, the user specifies
    a range of values and color the value of a channel is assigned linearly in that range, saturating at the limits.

    Args:

         pts: The points to assign colors to  of shape n_ps*[2 or 3]

         lims: The limits at which colors should saturate of shape [2 or 3]*2, where lims[i,0] gives the lower
         limit for dimension i and lims[i,1] gives the upper limit.

    Returns:

        clrs: The assigned colors of shape n_pts*4, where clrs[i,:] is the RGBA color for point i.
    """

    n_pts, n_dims = pts.shape

    x_lims = lims[0, :]
    y_lims = lims[1, :]
    x_range = x_lims[1] - x_lims[0]
    y_range = y_lims[1] - y_lims[0]

    r_vls = bound((pts[:,0] - x_lims[0])/x_range, lb=0, ub=1)
    b_vls = bound((pts[:,1] - y_lims[0])/y_range, lb=0, ub=1)

    if n_dims == 3:
        z_lims = lims[2, :]
        z_range = z_lims[1] - z_lims[0]
        g_vls = bound((pts[:, 2] - z_lims[0])/z_range, lb=0, ub=1)
    else:
        g_vls = np.zeros(n_pts)

    alpha_vls = np.ones(n_pts)

    return np.stack([r_vls, g_vls, b_vls, alpha_vls]).transpose()


def plot_three_dim_pts(pts: Union[torch.Tensor, np.ndarray], clrs: np.ndarray = None,
                       a: OptionalAxes = None) -> plt.Axes:
    """  Plots a 3-d cloud of points.

    Args:
        pts: The points to plot of shape n_smps*3

    Returns:
        a: The axes plotted into
    """

    if isinstance(pts, torch.Tensor):
        pts = pts.detach().cpu().numpy()

    if a is None:
        f = plt.figure()
        a = f.add_subplot(projection='3d')

    if clrs is None:
        a.scatter(pts[:,0], pts[:, 1], pts[:, 2], '.')
    else:
        a.scatter(pts[:,0], pts[:, 1], pts[:, 2], '.', c=clrs)
        #for pt_i, pt in enumerate(pts):
        #    a.scatter(pt[0], pt[1], pt[2], '.', color=clrs[pt_i])

    return a


def plot_torch_dist(mn_f, std_f, extra_title_str: str = None, axes: OptionalMultipleAxes = None, vis_dim: int = 0):
    """ Plots conditional mean and standard deviations across space.

     Args:

         mn_f: A torch module which computes a conditional mean given 2-d input

         std_f: A torch module which computes a conditional standard deviation given 2-d input

         axes: Axes to plot into.  axes[0] is for the mean and axes[1] is for standard deviation.

         vis_dim: If mn_f and std_f produce multi-d output, this is the particular dimention to visualize

     """

    if extra_title_str == None:
        extra_title_str = ''

    mn_im, _, _ = generate_2d_fcn_image(torch_mod_to_fcn(mn_f), vis_dim=vis_dim)
    std_im, _, _ = generate_2d_fcn_image(torch_mod_to_fcn(std_f), vis_dim=vis_dim)

    if axes is None:
        axes = [None]*2
        axes[0] = plt.subplot(1, 2, 1)
        axes[1] = plt.subplot(1, 2, 2)

    divider = make_axes_locatable(axes[0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    im = axes[0].imshow(mn_im)
    plt.colorbar(im, cax=cax, orientation='vertical')
    axes[0].set_title('Mean' + extra_title_str)

    divider = make_axes_locatable(axes[1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    im = axes[1].imshow(std_im)
    plt.colorbar(im, cax=cax, orientation='vertical')
    axes[1].set_title('Std' + extra_title_str)


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

