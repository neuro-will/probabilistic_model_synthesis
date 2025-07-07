""" Tools for simulating data for testing and developing models. """

import math
from typing import Optional, Sequence

import numpy as np
import torch

from janelia_core.math.basic_functions import pts_in_arc
from janelia_core.ml.extra_torch_modules import FixedOffsetAbs
from janelia_core.ml.extra_torch_modules import Unsqueeze
from janelia_core.ml.torch_distributions import CondMatrixProductDistribution
from janelia_core.ml.torch_distributions import CondGaussianDistribution
from janelia_core.ml.wandering_modules import SumOfBumpFcns


def generate_sum_of_bump_fcns_dist(n_bump_fcns: int, d_in: int, p: int, dim_ranges: Optional[np.ndarray] = None,
                                   bump_w: float = .1, mn_m_std: float = 1, std_m_std: float = .1) -> CondMatrixProductDistribution:
    """ Generates a conditional Gaussian distribution where the cond. mean and standard deviation are sum of bump fcns.

    More specifically, this represents a conditional Gaussian distribution over p dimensional random variables where:

        1) The mean for each dimension is a sum of bump functions (where the bump functions take d_in dimensional input)

        2) The standard deviation for each dimension is the absolute value of a sum of bump functions (where again the
        bump funcitons take d_in dimensional input) followed by a fixed offset (.01) to enforce that standard deviations
        are strictly positive.

    Args:

        n_bump_fcns: The number of individual functions in the sum

        d_in: The dimensionality of the input to the function

        p: The dimensionality of the variables the distribution is over

        dim_ranges: The range bump centers should span in each dimensions. dim_ranges[i,:] gives the min and max
        range for dimension i.  If None, all ranges will be [0, 1]

        bump_w: The width (standard deviation of Gaussian bumps) of each bump in each input dimension

        mn_m_std: The standard deviation to use when randomly assigning magnitudes from a centered Gaussian distribution
        to the bumps for the mean function

        std_m_std: The standard deviation to use when randomly assigning magnitudes from a centered Gaussian
        distribution to the bumps for the standard deviation function

    Returns:

        d: The generated distribution.
    """

    if dim_ranges is None:
        dim_ranges = np.asarray([[0, 1.0]]*p)

    # Generate random bump centers for means
    rnd_centers = np.random.uniform(size=[d_in, 2*n_bump_fcns])
    for d_i in range(p):
        rnd_centers[d_i, :] = rnd_centers[d_i, :]*(dim_ranges[d_i, 1] - dim_ranges[d_i, 0]) + dim_ranges[d_i, 0]
    rnd_centers = torch.tensor(rnd_centers, dtype=torch.float)

    mn_rnd_centers = rnd_centers[:, 0:n_bump_fcns]
    std_rnd_centers = rnd_centers[:, n_bump_fcns:]

    dists = [None]*p
    for d_i in range(p):
        mn_f = torch.nn.Sequential(SumOfBumpFcns(c=mn_rnd_centers,
                                                 w=bump_w*torch.ones([d_in, n_bump_fcns]),
                                                 m=mn_m_std*torch.randn(n_bump_fcns),
                                                 c_bounds=None, w_bounds=None),
                                   Unsqueeze(-1))

        std_f = torch.nn.Sequential(SumOfBumpFcns(c=std_rnd_centers,
                                                  w=bump_w*torch.ones([2, n_bump_fcns]),
                                                  m=std_m_std*torch.randn(n_bump_fcns),
                                                  c_bounds=None, w_bounds=None),
                                    FixedOffsetAbs(.01),
                                    Unsqueeze(-1))

        dists[d_i] = CondGaussianDistribution(mn_f=mn_f, std_f=std_f)

    return CondMatrixProductDistribution(dists=dists)


class IncreasingSinFcn(torch.nn.Module):
    """ Implements the function x + sin(3x) """

    def forward(self, x) -> torch.Tensor:
        return x + torch.sin(3*x)


def cone_and_projected_interval_sample(n_smps: int, locs: np.ndarray, ctr: np.ndarray, ang_range: Sequence[float],
                                       w: np.ndarray, interval: Sequence[float], big_std: float,
                                       small_std: float) -> np.ndarray:
    """ Samples variables in space so those within a arc have larger values and projected data falls in a interval.

    Args:

        n_smps: The number of samples to generate

        locs: The locations of variables in space, of shape n_smps*2

        ctr: The position of the origin for defining arcs in space

        ang_range: Variables within an arc defined by (ang_range[0], ang_range[1]) will be sampled with larger
        variances while those outside will be sampled with smaller variances.  Units should be in radians.

        w: The vector to use when projecting data, of shape n_smps

        interval: The interval that projected values should fall within.  Values in the range [interval[0], interval[1])
        will be accepted.

        big_std: The standard deviation of variables within the arc

        small_std: The standard deviation of variables outside of the arc

    Returns:

        smps: Samples of shape n_smps*d_x.

    """

    w = w.squeeze()  # Make sure w is just a vector

    # Determine which variables are within the arc and which are outside
    loc_centered = locs - ctr
    angs = np.asarray([math.atan2(v[0], v[1]) for v in loc_centered])
    angs[angs < 0] += 2*np.pi
    big_vars = np.logical_and(angs >= ang_range[0], angs < ang_range[1])
    small_vars = np.logical_not(big_vars)

    # Generate samples here
    raw_dim = len(w)
    smps = np.zeros([n_smps, raw_dim])
    smps[:] = np.nan

    n_accepted_smps = 0
    while n_accepted_smps < n_smps:
        # Determine which samples we still need to fill in
        needed_smps = np.argwhere(np.isnan(smps[:, 0])).squeeze(axis=1)
        n_needed_smps = len(needed_smps)

        cand_smps = np.random.randn(n_needed_smps, raw_dim)
        cand_smps[:, big_vars] = big_std*cand_smps[:, big_vars]
        cand_smps[:, small_vars] = small_std*cand_smps[:, small_vars]
        cand_projs = np.sum(cand_smps*w, 1)
        keep_smps = np.argwhere(np.logical_and(cand_projs >= interval[0], cand_projs < interval[1])).squeeze(axis=1)
        n_keep_smps = len(keep_smps)

        smps[needed_smps[0:n_keep_smps], :] = cand_smps[keep_smps, :]
        n_accepted_smps = np.sum(np.logical_not(np.isnan(smps[:, 0])))

    return smps


def efficient_cone_and_projected_interval_sample(n_smps: int, locs: torch.tensor, ctr: torch.tensor,
                                                 ang_range: Sequence[float], w: torch.tensor, interval: Sequence[float],
                                                 big_std: float, small_std: float,
                                                 device:torch.device = None) -> np.ndarray:

    """ Samples variables in space so those within a arc have larger values and projected data falls in a interval.

    ** Note the distribution of sample points for this function is NOT the same as the function
    cone_and_projected_interval_sample.

    Args:

        n_smps: The number of samples to generate

        locs: The locations of variables in space, of shape n_smps*2

        ctr: The position of the origin for defining arcs in space

        ang_range: Variables within an arc defined by (ang_range[0], ang_range[1]) will be sampled with larger
        variances while those outside will be sampled with smaller variances.  Units should be in radians.

        w: The vector to use when projecting data, of shape n_smps

        interval: The interval that projected values should fall within.

        big_std: The standard deviation of variables within the arc

        small_std: The standard deviation of variables outside of the arc

    Returns:

        smps: Samples of shape n_smps*d_x.

    """

    if device is None:
        device = torch.device('cpu')

    # Make sure w is just a vector and put it on the right device
    w = w.squeeze().to(device)

    # ================================================================================
    # Determine which variables are within the arc and which are outside
    # ================================================================================
    big_vars = pts_in_arc(pts=np.asarray(locs), ctr=np.asarray(ctr), arc_angle=ang_range)
    big_vars = torch.tensor(big_vars, device=device)
    small_vars = ~big_vars

    # ================================================================================
    # Generate samples here
    # ================================================================================
    with torch.no_grad():
        big_w = w[big_vars] # Portion of w for variables with large variance
        small_w = w[small_vars] # Portion of w for variables with small variance
        n_big_vars = len(big_w)
        n_small_vars = len(small_w)

        # Get a unit vector pointing in the same direction as the portion of w for the variables with large std
        big_var_l = torch.sqrt(torch.sum(big_w**2))
        big_var_unit_w = big_w/big_var_l

        # Generate the samples for the variables with small variance
        small_smps = small_std*torch.randn([n_smps, n_small_vars], device=device)

        # Generate the random value we want the data to project to for all samples
        interval_span = interval[1] - interval[0]
        interval_tgts = interval_span*torch.rand(n_smps, device=device) + interval[0]
        big_var_tgts = interval_tgts - torch.matmul(small_smps, small_w)

        # Generate values for big variables along the direction of big_w that will project to the target values
        big_var_base = torch.tile(big_var_unit_w, [n_smps, 1])
        big_var_base = big_var_base*big_var_tgts.unsqueeze(1)/big_var_l

        # Generate noise in a direction orthogonal to big_w for the variables with big std
        big_var_orth_noise = big_std*torch.randn([n_smps, n_big_vars], device=device)
        noise_projs = torch.matmul(big_var_orth_noise, big_var_unit_w)
        noise_projs = torch.tile(big_var_unit_w, [n_smps, 1])*noise_projs.unsqueeze(1)
        big_var_orth_noise = big_var_orth_noise - noise_projs

        # Generate the final samples for the variables with big standard deviation
        big_smps = big_var_base + big_var_orth_noise

        # Put samples for variables with big and small std together
        smps = torch.zeros([n_smps, n_big_vars + n_small_vars], device=device)
        smps[:, big_vars] = big_smps
        smps[:, small_vars] = small_smps
        return smps
