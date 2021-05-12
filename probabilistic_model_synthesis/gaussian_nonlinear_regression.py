""" Tools for performing model synthesis over non-linear regression models with Gaussian noise. """

import torch
from typing import Optional, Sequence, Union

from janelia_core.ml.torch_distributions import CondVAEDistribution

# Define type aliases
OptionalDevices = Optional[Sequence[torch.device]]
OptionalDict = Optional[dict]
OptionalDistribution = Optional[CondVAEDistribution]
OptionalTensor = Optional[torch.Tensor]


class GNLDRMdl(torch.nn.Module):
    """ A nonlinear dimensionality regression model with Gaussian noise on predicted variables.

    A Gaussian nonlinear regression (GNLR) model can be described by the following generative process, conditioned
    on input x_t \in R^n:

    First, mean values of predicted variables are generated according to:

        z_t = w_in*x_t + b_in, for w_in \in R^{p \times n} and b_in \in R^p
        l_t = m(z_t), l_t \in R^q and m() is a neural network
        mn_t = w_out*l_t + b_out, for w_out, b_out \in R^q, where * only here denotes the element-wise product

    Observed variables, y_t \in R^q are then generated according to:

        y_t|mn_t ~ N(mn_t, \Psi), for a diagonal PSD matrix \Psi \in R^{q \times q}

    This object by default will use it's own internal parameters, but it's various functions allow a user
    to provide their own parameter values, making it easy to use these objects when fitting models and
    wanting point estimates for some parameters (which would be represented in the internal parameters of this
    object) while treating other parameters as random variables.
    """

    def __init__(self, m: torch.nn.Module, w_in: OptionalTensor = None, b_in: OptionalTensor = None,
                 w_out: OptionalTensor = None, b_out: OptionalTensor = None):
        """ Creates a new GNLRMdl Object.

        Note: when creating this object the user can supply values for the parameters or set any of these to None.
        If they are set to None, no parameters will be created for them in the model.  In this case, it will be expected
        that values for them will be provided when calling any function (e.g., sample) that need these values.  The
        reason for this is that model objects may be used in frameworks which treat some/all of the parameters of a
        model as random variables, and we will want to provide sampled values of these parameters when working with the
        models, so there is no point in taking up memory representing them within the model object itself.

        Args:
            m: The mapping from the low-d latent space to the higher-d latent space.

            w_in: The weights projecting into the low-d space

            b_in: The biases for projecting into the low-d space

            w_out: The weights for projecting into the means of predicted variables

            b_out: The biases for projecting into the means of predicted variables

        """
        super().__init__()

        self.m = m

        if w_in is not None:
            self.w_in = torch.nn.Parameter(w_in)
        else:
            self.w_in = None

        if b_in is not None:
            self.b_in = torch.nn.Parameter(b_in)
        else:
            self.b_in = None

        if w_out is not None:
            self.w_out = torch.nn.Parameter(w_out)
        else:
            self.w_out = None

        if b_out is not None:
            self.b_out = torch.nn.Parameter(b_out)
        else:
            self.b_out = None

    def cond_log_prob(self, z: torch.Tensor, x: torch.Tensor, lm: OptionalTensor = None, mn: OptionalTensor = None,
                      psi: OptionalTensor = None, s: OptionalTensor = None) -> torch.Tensor:
        """ Computes the log probability of observations given latents.

        Args:
            z: Latents of shape n_smps*n_latent_dims

            x: Observed values of shape n_smps*n_obs_variables

            lm: If provided, uses this in place of the loading matrix parameter of the model.

            mn: If provided, uses this in place of the mean parameter of the model.

            psi: If provided, uses this in place of the psi parameter of the model.

            s: If provided, use this in place of the scales parameters of the model.

        Returns:

            ll: The log-likelihood for each sample.  Of shape n_smps.

        """

        n_smps, n_obs_vars = x.shape

        if lm is None:
            lm = self.lm
        if mn is None:
            mn = self.mn
        if psi is None:
            psi = self.psi
        if s is None:
            s = self.s

        mns = torch.abs(s)*(torch.matmul(self.m(z), lm.T) + mn)

        ll = -.5*torch.sum(((x - mns)**2)/psi, dim=1)
        ll -= .5*torch.sum(torch.log(psi))
        ll -= .5*n_obs_vars*self.log_2_pi

        return ll

    def sample(self, n_smps: int, lm: OptionalTensor = None, mn: OptionalTensor = None,
               psi: OptionalTensor = None, s:OptionalTensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Generates samples from the model.

        Args:

            lm: If provided, uses this in place of the loading matrix parameter of the model.

            mn: If provided, uses this in place of the mean parameter of the model.

            psi: If provided, uses this in place of the psi parameter of the model.

            s: If provided, use this in place of the scales parameters of the model.

        Returns:

            z: Latent values of shape n_smps*n_latent_dims

            x: Observed values of shape n_smps*n_obs_variables

        """

        if lm is None:
            lm = self.lm
        if mn is None:
            mn = self.mn
        if psi is None:
            psi = self.psi
        if s is None:
            s = self.s

        n_obs_vars = lm.shape[0]

        z = torch.randn(n_smps, self.n_latent_vars)

        x_mn = torch.abs(s)*(torch.matmul(self.m(z), lm.T) + mn)
        x_noise = torch.randn(n_smps, n_obs_vars)*torch.sqrt(psi)
        x = x_mn + x_noise

        return z, x
