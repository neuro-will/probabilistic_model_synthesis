""" A collection of distributions that can be used with Pytorch.

Note: The distribution objects defined here are *not* subclasses of the torch.distributions.

"""

import torch

from janelia_core.ml.extra_torch_modules import IndSmpConstantBoundedFcn
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

    def __init__(self, n_vars: int, alpha_lb: float = 1.0, alpha_iv: float = 5.0,
                 beta_lb: float = .00001, beta_iv: float = 5.0):
        """ Creates a new FoldedNormalProductDistribution object.

        Args:

            int: The number of random variables the distribution is over.

            alpha_lb: The lower bound that alpha parameters can take on

            alpha_iv: The initial value for alpha parameters.  All distributions will be initialized to have the
            same initial values.

            beta_lb: The lower bound that beta parameters can take on

            beta_iv: The initial value for beta parameters.  All distributions will be initialized to have the
            same initial values.
        """

        self.n_vars = n_vars
        self.alpha_lb = alpha_lb
        self.beta_lb = beta_lb

        alpha_f = torch.nn.Sequential(IndSmpConstantRealFcn(n=n_vars), FixedOffsetAbs(o=alpha_lb))
        beta_f = torch.nn.Sequential(IndSmpConstantRealFcn(n=n_vars), FixedOffsetAbs(o=beta_lb))

        alpha_f[0].f.vl.data[:] = alpha_iv
        beta_f[0].f.vl.data[:] = beta_iv

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

        new_alpha = (mean**2)/(cur_std**2) - self.alpha_lb
        new_beta = mean/(cur_std**2) - self.beta_lb

        if torch.any(new_alpha < self.alpha_lb):
            raise(RuntimeError('Setting the specified mean requires one or more alpha values that are out of range.'))

        if torch.any(new_beta < self.beta_lb):
            raise(RuntimeError('Setting the specified mean requires one or more beta values that are out of range.'))

        self.conc_f[0].f.vl.data = new_alpha
        self.rate_f[0].f.vl.data = new_beta

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

        new_alpha = (cur_mn**2)/(std**2) - self.alpha_lb
        new_beta = cur_mn/(std**2) - self.beta_lb

        if torch.any(new_alpha < self.alpha_lb):
            raise(RuntimeError('Setting the specified std requires one or more alpha values that are out of range.'))

        if torch.any(new_beta < self.beta_lb):
            raise(RuntimeError('Setting the specified std requires one or more beta values that are out of range.'))

        self.conc_f[0].f.vl.data = new_alpha
        self.rate_f[0].f.vl.data = new_beta

