""" Tools for simulating data for testing and developing models. """

import math
from typing import Optional, Sequence, Tuple

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
    two_pi = 2*np.pi

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















def ss_sample_within_projected_interval(n_smps: int, w: np.ndarray, u: np.ndarray,
                                        interval: Sequence[float]) -> np.ndarray:
    """ Randomly generates data in a high-d subspace that when projected along a vectors falls within given bounds.

    Specifically, this function generates samples z_t, x_t such that

        z_t ~ N(0, I), z_i \in R^m

        w'U z_t \in [l, u), U \in R^{n \times m}, w \in R^n

        x_t = U z_t

        where l and u and lower and bounds.

    Args:

        n_smps: The number of samples to generate

        w: The vector to project x_t along

        U: The matrix defining the subspace to generate x_t samples in.

        interval: interval[0] is the lower bound and interval[1] is upper bound projected samples should fall within.

    Returns:

        x: The random samples x of shape n_smps*d_x

        z: The random z values that generated each sample of shape n_smps*d_z

    """
    w = w.squeeze() # Make sure w is just a vector

    latent_dim = u.shape[1]
    raw_dim = len(w)
    smps = np.zeros([n_smps, raw_dim])
    smps_z = np.zeros([n_smps, latent_dim])
    smps[:] = np.nan

    u_t = u.transpose()

    n_accepted_smps = 0
    while n_accepted_smps < n_smps:
        # Determine which samples we still need to fill in
        needed_smps = np.argwhere(np.isnan(smps[:, 0])).squeeze(axis=1)
        n_needed_smps = len(needed_smps)

        cand_z = np.random.randn(n_needed_smps, latent_dim)
        cand_smps = np.matmul(cand_z, u_t)
        cand_projs = np.sum(cand_smps*w, 1)
        keep_smps = np.argwhere(np.logical_and(cand_projs >= interval[0], cand_projs < interval[1])).squeeze(axis=1)
        n_keep_smps = len(keep_smps)

        smps[needed_smps[0:n_keep_smps], :] = cand_smps[keep_smps, :]
        smps_z[needed_smps[0:n_keep_smps], :] = cand_z[keep_smps, :]

        n_accepted_smps = np.sum(np.logical_not(np.isnan(smps[:, 0])))

    return smps, smps_z


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


def calc_mean_and_std_in_grid(locs: np.ndarray, vls: np.ndarray, grid_dims: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """ Calculates the mean and standard deviations of values associated with points in differents parts of a grid.

    Args:

        locs: The location of each point, of shape n_pts*2

        vls: The values of each point, of length n_pts

        grid_dims: The dimensions of the grid.  grid_dims[i,:] is the dimensions for dimension i and is of
        the form (start_vl, stop_vl, step_size).

    Returns:

        mns: Provides the mean in each grid division.  The first dimension corresponds to the first dimension of
        grid_dims, and the second the second.

        stds: Provides the standard deviation in each grid division.
    """

    div_0 = np.arange(*grid_dims[0, :])
    div_1 = np.arange(*grid_dims[1, :])

    n_div_0 = len(div_0) - 1
    n_div_1 = len(div_0) - 1

    mns = np.zeros([n_div_0, n_div_1])
    stds = np.zeros([n_div_0, n_div_1])
    mns[:] = np.nan
    stds[:] = np.nan

    for i_0 in range(n_div_0):
        div_0_lb = div_0[i_0]
        div_0_ub = div_0[i_0 + 1]
        div_0_pts = np.logical_and(locs[:, 0] >= div_0_lb, locs[:, 0] < div_0_ub)
        for i_1 in range(n_div_1):
            div_1_lb = div_1[i_1]
            div_1_ub = div_1[i_1 + 1]
            div_1_pts = np.logical_and(locs[:, 1] >= div_1_lb, locs[:, 1] < div_1_ub)
            div_pts = np.logical_and(div_0_pts, div_1_pts)
            mns[i_0, i_1] = np.mean(vls[div_pts])
            stds[i_0, i_1] = np.std(vls[div_pts])

    return mns, stds

