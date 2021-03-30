""" A collection of distributions that can be used with Pytorch.

Note: The distribution objects defined here are *not* subclasses of the torch.distributions.

"""

import torch

from janelia_core.ml.extra_torch_modules import IndSmpConstantBoundedFcn
from janelia_core.ml.torch_distributions import CondFoldedNormalDistribution


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

    def __init__(self, n_vars: int, mu_lb: float = 0.0, mu_ub: float = 10.0, mu_iv: float = 1.0,
                 sigma_lb: float = .001, sigma_ub: float = 10.0, sigma_iv: float = 1.0):
        """ Creates a new FoldedNormalProductDistribution object.

        Args:

            int: The number of random variables the distribution is over.

            mu_lb: The lower bound that mu parameters can take on

            mu_ub: The upper bound that mu parameters can take on

            mu_iv: The initial value for mu parameters.  All distributions will be initialized to have the
            same initial values.

            sigma_lb: The lower bound that sigma parameters can take on

            sigma_ub: The upper bound that sigma parameters can take on

            sigma_iv: The initial value for sigma parameters.  All distributions will be initialized to have the
            same initial values.
        """
        mu_f = IndSmpConstantBoundedFcn(n=n_vars, lower_bound=mu_lb, upper_bound=mu_ub, init_value=mu_iv)
        sigma_f = IndSmpConstantBoundedFcn(n=n_vars, lower_bound=sigma_lb, upper_bound=sigma_ub, init_value=sigma_iv)

        super().__init__(mu_f=mu_f, sigma_f=sigma_f)




