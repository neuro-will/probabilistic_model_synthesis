""" Tools for performing model synthesis over non-linear regression models with Gaussian noise. """

import math
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

        z_t = s_in \had_prod w_in*x_t + b_in, for w_in \in R^{p \times n}, b_in \in R^p, s_in \in R^p where * denotes
                                              matrix product and \had_prod denotes element-wise product

        l_t = m(z_t), l_t \in R^q and m() is a neural network

        mn_t = s_out \had_prod l_t + b_out, for s_out \in R^q, b_out \in R^q,

    Observed variables, y_t \in R^q are then generated according to:

        y_t|mn_t ~ N(mn_t, \Psi), for a diagonal PSD matrix \Psi \in R^{q \times q}

    This object by default will use it's own internal parameters, but it's various functions allow a user
    to provide their own parameter values, making it easy to use these objects when fitting models and
    wanting point estimates for some parameters (which would be represented in the internal parameters of this
    object) while treating other parameters as random variables.
    """

    def __init__(self, m: torch.nn.Module, w_in: OptionalTensor = None, s_in: OptionalTensor = None,
                 b_in: OptionalTensor = None, s_out: OptionalTensor = None, b_out: OptionalTensor = None,
                 psi: OptionalTensor = None):
        """ Creates a new GNLRMdl Object.

        Note: when creating this object the user can supply values for the parameters or set any of these to None.
        If they are set to None, no parameters will be created for them in the model.  In this case, it will be expected
        that values for them will be provided when calling any function (e.g., sample) that need these values.  The
        reason for this is that model objects may be used in frameworks which treat some/all of the parameters of a
        model as random variables, and we will want to provide sampled values of these parameters when working with the
        models, so there is no point in taking up memory representing them within the model object itself.

        Args:
            m: The mapping from the low-d latent space to the higher-d latent space.  Should accept input with a
            dimensionality equal to p and the output dimensionality should be equal to the number of predicted variables

            w_in: The weights projecting into the low-d space.  Of shape n_input_vars*p

            s_in: The scales for projecting into the low-d space. Of shape p

            b_in: The biases for projecting into the low-d space. Of shape p

            s_out: The scales for mapping to the the means of predicted variables. Of shape n_predicted_variables

            b_out: The biases for projecting into the means of predicted variables. Of shape n_predicted variables

            psi: A vector of noise variances of for the predicted variables

        """
        super().__init__()

        self.m = m

        if w_in is not None:
            self.w_in = torch.nn.Parameter(w_in)
        else:
            self.w_in = None

        if s_in is not None:
            self.s_in = torch.nn.Parameter(s_in)
        else:
            self.s_in = None

        if b_in is not None:
            self.b_in = torch.nn.Parameter(b_in)
        else:
            self.b_in = None

        if s_out is not None:
            self.s_out = torch.nn.Parameter(s_out)
        else:
            self.w_out = None

        if b_out is not None:
            self.b_out = torch.nn.Parameter(b_out)
        else:
            self.b_out = None

        if psi is not None:
            self.psi = torch.nn.Parameter(psi)
        else:
            self.psi = None

        self.register_buffer('log_2_pi', torch.log(torch.tensor(2 * math.pi)))

    def cond_log_prob(self, x: torch.Tensor, y: torch.Tensor, w_in: OptionalTensor = None, s_in: OptionalTensor = None,
                      b_in: OptionalTensor = None, s_out: OptionalTensor = None, b_out: OptionalTensor = None,
                      psi: OptionalTensor = None) -> torch.Tensor:
        """ Computes the log probability of predicted variables given input.

        Args:
            x: Input into the regression model of shape n_smps*n_input_variables

            y: Predicted values of shape n_smps*n_predicted_variables

            w_in: If provided, this is used in place of the w_in parameter of the model

            s_in: If provided, this is used in place of the s_in parameter of the model

            b_in: If provided, this is used in place of the b_in parameter of the model

            s_out: If provided, this is used in place of the s_out parameter of the model

            b_out: If provided, this is used in place of the b_out parameter of the model

            psi: If provided, this is used in place of the psi parameter of the model

        Returns:

            ll: The log-likelihood for each sample.  Of shape n_smps.

        """

        n_smps, n_pred_vars = y.shape

        if w_in is None:
            w_in = self.w_in
        if b_in is None:
            b_in = self.b_in
        if s_in is None:
            s_in = self.s_in
        if s_out is None:
            s_out = self.w_out
        if b_out is None:
            b_out = self.b_out
        if psi is None:
            psi = self.psi

        mns = self.cond_mean(x=x, w_in=w_in, s_in=s_in, b_in=b_in, s_out=s_out, b_out=b_out)

        ll = -.5*torch.sum(((y - mns)**2)/psi, dim=1)
        ll -= .5*torch.sum(torch.log(psi))
        ll -= .5*n_pred_vars*self.log_2_pi

        return ll

    def cond_mean(self, x: torch.Tensor, w_in: OptionalTensor = None, s_in: OptionalTensor = None,
                  b_in: OptionalTensor = None, s_out: OptionalTensor = None,
                  b_out: OptionalTensor = None) -> torch.Tensor:
        """ Computes mean of predicted variables given input.

        Args:
            x: Input into the regression model of shape n_smps*n_input_variables

            w_in: If provided, this is used in place of the w_in parameter of the model

            s_in: If provided, this is used in place of the s_in parameter of the model

            b_in: If provided, this is used in place of the b_in parameter of the model

            s_out: If provided, this is used in place of the s_out parameter of the model

            b_out: If provided, this is used in place of the b_out parameter of the model

        Returns:

            mn: Means of predicted variables conditioned on input. Of shape n_smps*n_predicted variables

        """
        if w_in is None:
            w_in = self.w_in
        if b_in is None:
            b_in = self.b_in
        if s_in is None:
            s_in = self.s_in
        if s_out is None:
            s_out = self.w_out
        if b_out is None:
            b_out = self.b_out

        z = s_in*torch.matmul(x, w_in) + b_in
        l = self.m(z)
        return s_out*l + b_out

    def sample(self, x: torch.Tensor, w_in: OptionalTensor = None, s_in: OptionalTensor = None,
               b_in: OptionalTensor = None, s_out: OptionalTensor = None, b_out: OptionalTensor = None,
               psi: OptionalTensor = None) -> torch.Tensor:
        """ Generates samples from the model.

        Args:

            x: Conditioning input to generate samples from. Of shape n_smps*n_input_variables

            w_in: If provided, this is used in place of the w_in parameter of the model

            s_in: If provided, this is used in place of the s_in parameter of the model

            b_in: If provided, this is used in place of the b_in parameter of the model

            s_out: If provided, this is used in place of the s_out parameter of the model

            b_out: If provided, this is used in place of the b_out parameter of the model

            psi: If provided, this is used in place of the psi parameter of the model

        Returns:

            y: Samples of shape n_smps*n_predicted_variables

        """

        if w_in is None:
            w_in = self.w_in
        if b_in is None:
            b_in = self.b_in
        if s_in is None:
            s_in = self.s_in
        if s_out is None:
            s_out = self.w_out
        if b_out is None:
            b_out = self.b_out
        if psi is None:
            psi = self.psi

        mns = self.cond_mean(x=x, w_in=w_in, s_in=s_in, b_in=b_in, s_out=s_out, b_out=b_out)
        noise = torch.randn(mns.shape)*torch.sqrt(psi)

        return mns + noise


