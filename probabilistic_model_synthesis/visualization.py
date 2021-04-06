""" Tools for visualizing data and the results of modeling fitting. """

from typing import Callable, Sequence, Union

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from janelia_core.visualization.image_generation import generate_2d_fcn_image
from janelia_core.ml.utils import torch_mod_to_fcn

# Define aliases
OptionalAxes = Union[plt.Axes, None]
OptionalCallable = Union[Callable, None]
OptionalMultipleAxes = Union[Sequence[plt.Axes], None]


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