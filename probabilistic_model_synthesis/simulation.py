""" Tools for simulating data for testing and developing models. """

import math
from typing import Optional, Sequence

import numpy as np
import torch

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


def sample_proj_data_from_cone(n_smps: np.ndarray, w: np.ndarray, ang_range: Sequence[float]) -> np.ndarray:
    """ Generates samples from a standard normal, which when projected into 2-dimensions, fall within a cone region.

    Args:

        n_smps: The number of samples to generate

        w: Weights for projecting data

        ang_range: The range of angles (in radians) defining the cone region that projected samples need to fall into.

    """

    raw_dim, proj_dim = w.shape
    smps = np.zeros([n_smps, raw_dim])
    smps[:] = np.nan

    n_accepted_smps = 0
    while n_accepted_smps < n_smps:

        # Determine which samples we still need to fill in
        needed_smps = np.argwhere(np.isnan(smps[:, 0])).squeeze(axis=1)

        n_needed_smps = len(needed_smps)

        # Generate candidate samples
        cand_smps = np.random.randn(n_needed_smps, raw_dim)
        cand_projs = np.matmul(cand_smps, w)
        cand_angs = np.asarray([math.atan2(v[0], v[1]) for v in cand_projs])
        keep_smps = np.argwhere(np.logical_and(cand_angs > ang_range[0], cand_angs < ang_range[1])).squeeze(axis=1)
        n_keep_smps = len(keep_smps)

        smps[needed_smps[0:n_keep_smps], :] = cand_smps[keep_smps, :]

        n_accepted_smps = np.sum(np.logical_not(np.isnan(smps[:, 0])))

    return smps


def sample_proj_data_from_interval(n_smps: np.ndarray, w: np.ndarray, interval: Sequence[float]) -> np.ndarray:
    """ Generates random samples from a standard normal that fall within a projected range.

    Args:

        n_smps: The number of samples to produce

        w: The vector to project samples onto

        interval: The interval that projections must fall into.  The acceptance range will be of
        of the form [interval[0], interval[1]).

    Returns:

        smps: The generated samples.  Each row is a sample.
    """

    w = w.squeeze() # Make sure w is just a vector

    raw_dim = len(w)
    smps = np.zeros([n_smps, raw_dim])
    smps[:] = np.nan

    n_accepted_smps = 0
    while n_accepted_smps < n_smps:
        # Determine which samples we still need to fill in
        needed_smps = np.argwhere(np.isnan(smps[:, 0])).squeeze(axis=1)
        n_needed_smps = len(needed_smps)

        cand_smps = np.random.randn(n_needed_smps, raw_dim)
        cand_projs = np.sum(cand_smps*w, 1)
        keep_smps = np.argwhere(np.logical_and(cand_projs >= interval[0], cand_projs < interval[1])).squeeze(axis=1)
        n_keep_smps = len(keep_smps)

        smps[needed_smps[0:n_keep_smps], :] = cand_smps[keep_smps, :]
        n_accepted_smps = np.sum(np.logical_not(np.isnan(smps[:, 0])))

    return smps

class QuadrantFcn(torch.nn.Module):
    """ Represents a 2-d function where the behavior in different quadrants of input space differs.

    Specifically, this is a function of x:R^2 -> y:R of the form:

        y = x[0]*x[1], x[0] > 0, x[1] > 0
        y = -x[0]*x[1], x[0] < 0, x[1] < 0
        y = 0, Otherwise
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Computes output from input. """

        n_smps = x.shape[0]
        y = torch.zeros(n_smps, 1)

        quad_0_pts = torch.logical_and(x[:, 0] > 0, x[:, 1] > 0)
        quad_1_pts = torch.logical_and(x[:, 0] < 0, x[:, 1] < 0)

        y[quad_0_pts] = x[quad_0_pts, 0:1]*x[quad_0_pts, 1:2]
        y[quad_1_pts] = -1*x[quad_1_pts, 0:1]*x[quad_1_pts, 1:2]

        return y


class QuadXOR(torch.nn.Module):
    """ A surface defined by z = XOR(x > x_0, y > y_0) * [a*(x - x_0)^2 + b*(y - y_0)^2]

    Intuitively, this is a function that multiplies an XOR and a quadratic surface, so only the portions of
    the quadratic surface in the off-diagonal quadrants of input space are non-zero.
    """

    def __init__(self, ctr: torch.Tensor, coefs: torch.Tensor):
        """ Creates a new QuadXOR module.

        Args:

            ctr: the vector [x_0, x_1]

            coefs: the vector [a, b]

        """

        super().__init__()

        self.ctr = torch.nn.Parameter(ctr)
        self.coefs = torch.nn.Parameter(coefs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Computes output from input.

        Args:
            x: Input of shape n_smps*2

        Returns:
            output: Of length n_smps*1
    """

        xor = torch.logical_xor(x[:, 0] > self.ctr[0], x[:, 1] > self.ctr[1])
        z = torch.sum(self.coefs*((x - self.ctr)**2), dim=1)
        return (xor*z).unsqueeze(-1)
