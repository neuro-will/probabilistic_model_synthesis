""" A collection of distributions that can be used with Pytorch.

Note: The distribution objects defined here are *not* subclasses of the torch.distributions.

"""

import numpy as np
import torch

from janelia_core.ml.extra_torch_modules import ElementWiseTanh
from janelia_core.ml.extra_torch_modules import IndSmpConstantRealFcn
from janelia_core.ml.extra_torch_modules import FixedOffsetAbs
from janelia_core.ml.torch_distributions import CondFoldedNormalDistribution
from janelia_core.ml.torch_distributions import CondGammaDistribution


class FoldedNormalProductDistribution(CondFoldedNormalDistribution):
    """ A multivariate distribution, formed as the product of independent Folded Normal distributions.

    Specifically, let X be a vector of random variables.  We model

    P(X) = \prod P_i(X_i; mu_i, sigma_i),

    where mu_i and sigma_i are the parameters of folder normal distribution for random variable X_i.

    Note: This object extends CondFoldedNormalDistribution, even though it does not represent conditional
    distributions (or equivalently, it represents the same distribution over all possible conditioning data). By
    extending CondFoldedNormalDistribution, this distribution can be used with code which provides conditioning input,
    though it should be understood, that while the interface of the various methods will accept conditioning input,
    this input will be ignored.
    """

    def __init__(self, n_vars: int, mu_lb: float = .01, mu_iv: float = .1,
                 sigma_lb: float = .001, sigma_iv: float = .1):
        """ Creates a new FoldedNormalProductDistribution object.

        Args:

            int: The number of random variables the distribution is over.

            mu_lb: The lower bound that mu parameters can take on

            mu_iv: The initial value for mu parameters.  All distributions will be initialized to have the
            same initial values.

            sigma_lb: The lower bound that sigma parameters can take on

            sigma_iv: The initial value for sigma parameters.  All distributions will be initialized to have the
            same initial values.
        """
        mu_f = torch.nn.Sequential(IndSmpConstantRealFcn(n=n_vars), FixedOffsetAbs(o=mu_lb))
        sigma_f = torch.nn.Sequential(IndSmpConstantRealFcn(n=n_vars), FixedOffsetAbs(o=sigma_lb))

        mu_f[0].f.vl.data[:] = mu_iv
        sigma_f[0].f.vl.data[:] = sigma_iv

        super().__init__(mu_f=mu_f, sigma_f=sigma_f)


class GammaProductDistribution(CondGammaDistribution):
    """ A multivariate distribution, formed as the product of independent Gamma distributions.

    Specifically, let X be a vector of random variables.  We model

    P(X) = \prod P_i(X_i; alpha_i, beta_i),

    where alpha_i and beta_i are the shape and rate parameters of a Gamma distribution.

    Note: This object extends CondGammaDistribution, even though it does not represent conditional
    distributions (or equivalently, it represents the same distribution over all possible conditioning data). By
    extending CondGammaDistribution, this distribution can be used with code which provides conditioning input,
    though it should be understood, that while the interface of the various methods will accept conditioning input,
    this input will be ignored.
    """

    def __init__(self, n_vars: int, alpha_lb: float = 1.0, alpha_ub: float = 1E3, alpha_iv: float = 5.0,
                 beta_lb: float = 1E-3, beta_ub: float = 1E5, beta_iv: float = 5.0):
        """ Creates a new FoldedNormalProductDistribution object.

        Args:

            int: The number of random variables the distribution is over.

            alpha_lb: The lower bound that alpha parameters can take on

            alpha_ub: The upper bound that alpha parameters can take on

            alpha_iv: The initial value for alpha parameters.  All distributions will be initialized to have the
            same initial values.

            beta_lb: The lower bound that beta parameters can take on

            beta_ub: The upper bound that beta parameters can take on

            beta_iv: The initial value for beta parameters.  All distributions will be initialized to have the
            same initial values.
        """

        self.n_vars = n_vars
        self.alpha_lb = alpha_lb
        self.alpha_ub = alpha_ub
        self.beta_lb = beta_lb
        self.beta_ub = beta_ub

        alpha_o = .5*(alpha_lb + alpha_ub)
        alpha_s = .5*(alpha_ub - alpha_lb)
        beta_o = .5*(beta_lb + beta_ub)
        beta_s = .5*(beta_ub - beta_lb)

        self.alpha_o = alpha_o
        self.alpha_s = alpha_s
        self.beta_o = beta_o
        self.beta_s = beta_s

        alpha_f = torch.nn.Sequential(IndSmpConstantRealFcn(n=n_vars), ElementWiseTanh(o=alpha_o, s=alpha_s))
        beta_f = torch.nn.Sequential(IndSmpConstantRealFcn(n=n_vars), ElementWiseTanh(o=beta_o, s=beta_s))

        alpha_iv_t = np.arctanh((alpha_iv - alpha_o) / alpha_s)
        beta_iv_t = np.arctanh((beta_iv - beta_o) / beta_s)

        alpha_f[0].f.vl.data[:] = alpha_iv_t
        beta_f[0].f.vl.data[:] = beta_iv_t

        super().__init__(conc_f=alpha_f, rate_f=beta_f)

    def set_mean(self, mean: torch.Tensor) -> torch.Tensor:
        """ Sets the means of the gamma distributions, while not adjusting standard deviations.

        This function adjusts the rate and concentration parameters of the individual Gamma distributions, to achieve
        the specified means while not adjusting the standard deviations.

        Args:
            mean: Tensor specifying mean values to set.  mean[i] is the mean for the distribution of the i^th
            random variable.

        Raises:
            ValueError: If the length of mean is different than the number of random variables the distribution is over.

            RuntimeError: If setting the specified means requires one or more alpha or beta values that are out of
            bounds.

        """

        if len(mean) != self.n_vars:
            raise(ValueError('Length of mean does not equal number of random variables distribution is specified over.'))

        cur_std = self.std(x=torch.ones(self.n_vars)).squeeze()

        new_alpha = (mean**2)/(cur_std**2)
        new_beta = mean/(cur_std**2)

        if torch.any((new_alpha < self.alpha_lb)) or torch.any((new_alpha > self.alpha_ub)):
            raise(RuntimeError('Setting the specified mean requires one or more alpha values that are out of range.'))

        if torch.any((new_beta < self.beta_lb)) or torch.any((new_beta > self.beta_ub)):
            raise (RuntimeError('Setting the specified mean requires one or more alpha values that are out of range.'))

        new_alpha_t = torch.arctanh((new_alpha - self.alpha_o)/self.alpha_s)
        new_beta_t = torch.arctanh((new_beta - self.beta_o)/self.beta_s)

        self.conc_f[0].f.vl.data = new_alpha_t
        self.rate_f[0].f.vl.data = new_beta_t

    def set_std(self, std: torch.Tensor) -> torch.Tensor:
        """ Sets the standard deviations of the gamma distributions, while not adjusting means.

        This function adjusts the rate and concentration parameters of the individual Gamma distributions, to achieve
        the specified standard deviations while not adjusting the means.

        Args:
            std: Tensor specifying standard deviation values to set.  stds[i] is the standard deviation for the
            distribution of the i^th random variable.

        Raises:
            ValueError: If the length of std is different than the number of random variables the distribution is over.

            RuntimeError: If setting the specified standard deviations requires one or more alpha or beta values that
            are out of bounds
        """

        if len(std) != self.n_vars:
            raise (
                ValueError('Length of std does not equal number of random variables distribution is specified over.'))

        cur_mn = self(x=torch.ones(self.n_vars)).squeeze()

        new_alpha = (cur_mn**2)/(std**2)
        new_beta = cur_mn/(std**2)

        if torch.any((new_alpha < self.alpha_lb)) or torch.any((new_alpha > self.alpha_ub)):
            raise(RuntimeError('Setting the specified mean requires one or more alpha values that are out of range.'))

        if torch.any((new_beta < self.beta_lb)) or torch.any((new_beta > self.beta_ub)):
            raise (RuntimeError('Setting the specified mean requires one or more alpha values that are out of range.'))

        new_alpha_t = torch.arctanh((new_alpha - self.alpha_o)/self.alpha_s)
        new_beta_t = torch.arctanh((new_beta - self.beta_o)/self.beta_s)

        self.conc_f[0].f.vl.data = new_alpha_t
        self.rate_f[0].f.vl.data = new_beta_t


class SampleLatentsGaussianVariationalPosterior(torch.nn.Module):
    """ Represents the posterior distribution over latent variables for each sample.

    When creating this object, the user needs to specify the number of data points we will be working with.  The
    object will then be initialized with parameters for the mean for the posterior over the latents for each data
    point (the covariance for all data points is the same).

    In many applications we may not want the posterior distributions over all data points, so many of the methods
    of this object ask the user to specify the indices of data points relevant to the underlying computation.
    """

    def __init__(self, n_latent_vars: int, n_smps: int):
        """ Creates a new SampleLatentsGaussianVariationalPosterior object.

        Args:

            n_latent_vars: Latent dimensionality

            n_smps: The number of samples we will calculate posteriors over

        """
        super().__init__()

        self.m = n_latent_vars
        self.n = n_smps

        self.mns = torch.nn.Parameter(torch.zeros(n_smps, n_latent_vars))
        self.c = torch.nn.Parameter(torch.diag(torch.ones(n_latent_vars)))

    def cov(self) -> torch.Tensor:
        """ Returns the covariance matrix for the posterior over any data point. """

        return self.c.mm(self.c.T)

    def kl_btw_standard_normal(self, inds):
        """ Computes KL divergence between posterior of latent state over a set of data points and the standard normal.

        Args:

            inds: Indices of data points we form the posterior of latent state over.

        Returns:

            kl: The kl divergence for the posterior relative to the standard normal
        """

        n_kl_data_pts = len(inds)

        cov_m = self.cov()

        cov_trace_sum = n_kl_data_pts * torch.trace(cov_m)
        m_norm_sum = torch.sum(self.mns[inds] ** 2)
        m_sum = n_kl_data_pts * self.m
        log_det_sum = n_kl_data_pts * torch.logdet(cov_m)

        return .5 * (cov_trace_sum + m_norm_sum - m_sum - log_det_sum)

    def kl_btw_diagonal_normal(self, inds: torch.Tensor, mn_1: torch.Tensor, cov_1: torch.Tensor):
        """ Computes KL between posterior of latent state over a set of data points and normal with diagonal covariance.

        Args:

            inds: Indices of data points we form the posterior of latent state over.

            mn_1: The mean of the multivariate normal we compare to

            cov_1: Entries along the diagonal of the covariance matrix of the normal we compare to.  Should be a
            1-d tensor.

        Returns:

            kl: The kl divergence for the posterior relative to the standard normal
        """

        n_kl_data_pts = len(inds)

        cov_m = self.cov()

        cov_trace_sum = n_kl_data_pts*torch.trace(cov_m/cov_1)
        m_norm_sum = torch.sum(((self.mns[inds] - mn_1)**2)/cov_1)
        log_det_sum_1 = n_kl_data_pts*torch.sum(torch.log(cov_1))
        log_det_sum_0 = n_kl_data_pts*torch.logdet(cov_m)
        m_sum = n_kl_data_pts * self.m

        return .5 * (cov_trace_sum + m_norm_sum - m_sum + log_det_sum_1 - log_det_sum_0)

    def sample(self, inds: torch.Tensor) -> torch.Tensor:
        """ Samples latent values from the posterior for given data points.

        Args:

            inds: The indices of latents to return samples for.  Of type long

        Returns:

            samples: samples[i,:] is the sample for inds[i]

        """

        n_data_pts = len(inds)

        return torch.randn(n_data_pts, self.m, device=self.c.device).mm(self.c.T) + self.mns[inds, :]
