""" Classes and functions for performing model synthesis over factor analysis models. """

import copy
import itertools
import math
from typing import List, Tuple, Sequence, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim

from janelia_core.math.basic_functions import optimal_orthonormal_transform
from janelia_core.ml.extra_torch_modules import FixedOffsetAbs
from janelia_core.ml.extra_torch_modules import FixedOffsetTanh
from janelia_core.visualization.matrix_visualization import cmp_n_mats
from janelia_core.ml.torch_distributions import CondGaussianDistribution
from janelia_core.ml.torch_distributions import CondFoldedNormalDistribution
from janelia_core.ml.torch_distributions import CondVAEDistribution
from janelia_core.ml.torch_distributions import MatrixGaussianProductDistribution
from probabilistic_model_synthesis.distributions import CondGammaDistribution
from probabilistic_model_synthesis.distributions import GammaProductDistribution
from probabilistic_model_synthesis.utilities import enforce_floor
from probabilistic_model_synthesis.utilities import get_lr
from probabilistic_model_synthesis.utilities import get_scalar_vl
from probabilistic_model_synthesis.utilities import list_to_str


# Define type aliases
OptionalDict = Union[dict, None]
OptionalDistribution = Union[CondVAEDistribution, None]
OptionalTensor = Union[torch.Tensor, None]


class FAMdl(torch.nn.Module):
    """ A factor analysis model.

    Enables sampling from FA models as well as calculating likelihoods.

    This object by default will use it's own internal parameters, but it's various functions allow a user
    to provide their own parameter values, making it easy to use these objects when fitting FA models and
    wanting point estimates for some parameters (which would be represented in the internal parameters of this
    object) while treating other parameters as random variables.
    """

    def __init__(self, lm: OptionalTensor = None, mn: OptionalTensor = None, psi: OptionalTensor = None):
        """ Creates a new FA Object.

        Note: when creating this object the user can supply values for the loading matrix, mean vector or private
        variance or optionally set any of these to None.  If they are set to None, no parameters will be created for
        them in the model.  In this case, it will be expected that values for them will be provided when calling any
        function (e.g., sample, log_likelihood) that need these values.  The reason for this is that FA objects may
        be used in frameworks which treat some/all of the parameters of an FA model as random variables, and we will
        want to provide sampled values of these parameters when working with the models, so there is no point in
        taking up memory representing them within the FA object itself.

        Args:

            lm: The loading matrix of shape n_obs_variables*n_latent_dims

            mn: The mean vector of shape n_obs_variables

            psi: The private variances of shape n_obs_variables

        """
        super().__init__()

        if lm is not None:
            self.lm = torch.nn.Parameter(lm)
        else:
            self.lm = None

        if mn is not None:
            self.mn = torch.nn.Parameter(mn)
        else:
            self.mn = None

        if psi is not None:
            self.psi = torch.nn.Parameter(psi)
        else:
            self.psi = None

        self.register_buffer('log_2_pi', torch.log(torch.tensor(2*math.pi)))

    @staticmethod
    def compare_models(m1, m2):
        """ Visually compares two models.
        This model will display loading matrices rotated to best match each other.
        Args:
            m1: The fist model
            m2: The second model
        """

        ROW_SPAN = 24  # Number of rows in the gridspec
        COL_SPAN = 24  # Number of columns in the gridspec

        m2 = copy.deepcopy(m2)  # Copy m2 to local variable so changes to weights don't affect the passed object

        grid_spec = matplotlib.gridspec.GridSpec(ROW_SPAN, COL_SPAN)

        def _make_subplot(loc, r_span, c_span, d1, d2, title):
            subplot = plt.subplot(grid_spec.new_subplotspec(loc, r_span, c_span))

            min_vl = np.min(np.concatenate([d1, d2]))
            max_vl = np.max(np.concatenate([d1, d2]))

            identity_pts = np.asarray([min_vl, max_vl])
            plt.plot(identity_pts, identity_pts, 'k--')
            plt.xlabel('Mdl 1')
            plt.ylabel('Mdl 2')

            subplot.plot(d1, d2, 'r.')
            subplot.axis('equal')
            subplot.title = plt.title(title)

        # Make plots of scalar variables
        _make_subplot([0, 0], 8, 8, m1.mn.cpu().detach().numpy(), m2.mn.cpu().detach().numpy(), 'Mean')
        _make_subplot([12, 0], 8, 8, m1.psi.cpu().detach().numpy(), m2.psi.cpu().detach().numpy(), 'Psi')

        # Make plots of loading matrices
        c1 = m1.lm.cpu().detach().numpy()
        c2 = m2.lm.cpu().detach().numpy()
        print(c2.shape)
        o = optimal_orthonormal_transform(c1, c2)
        c2 = np.matmul(c2, o)
        c_diff = c1 - c2

        w1_grid_info = {'grid_spec': grid_spec}
        w1_cell_info = list()
        w1_cell_info.append({'loc': [0, 12], 'rowspan': 24, 'colspan': 3})
        w1_cell_info.append({'loc': [0, 16], 'rowspan': 24, 'colspan': 3})
        w1_cell_info.append({'loc': [0, 20], 'rowspan': 24, 'colspan': 3})
        w1_grid_info['cell_info'] = w1_cell_info
        cmp_n_mats([c1, c2, c_diff], show_colorbars=True, titles=['LM 1', 'LM 2', 'LM 1 - LM 2'],
                   grid_info=w1_grid_info)

    def cond_log_prob(self, z: torch.Tensor, x: torch.Tensor, lm: OptionalTensor = None, mn: OptionalTensor = None,
                      psi: OptionalTensor = None) -> torch.Tensor:
        """ Computes the log probability of observations given latents.

        Args:
            z: Latents of shape n_smps*n_latent_dims

            x: Observed values of shape n_smps*n_obs_variables

            lm: If provided, uses this in place of the loading matrix parameter of the model.

            mn: If provided, uses this in place of the mean parameter of the model.

            psi: If provided, uses this in place of the psi parameter of the model.

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

        mns = torch.matmul(z, lm.T) + mn

        ll = -.5*torch.sum(((x - mns)**2)/psi, dim=1)
        ll -= .5*torch.sum(torch.log(psi))
        ll -= .5*n_obs_vars*self.log_2_pi

        return ll

    def sample(self, n_smps: int, lm: OptionalTensor = None, mn: OptionalTensor = None,
               psi: OptionalTensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Generates samples from the model.

        Args:

            lm: If provided, uses this in place of the loading matrix parameter of the model.

            mn: If provided, uses this in place of the mean parameter of the model.

            psi: If provided, uses this in place of the psi parameter of the model.

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

        n_obs_vars, n_latents = lm.shape

        z = torch.randn(n_smps, n_latents)

        x_mn = torch.matmul(z, lm.T) + mn
        x_noise = torch.randn(n_smps, n_obs_vars)*torch.sqrt(psi)
        x = x_mn + x_noise

        return z, x


class FAVariationalPosterior(torch.nn.Module):
    """ Represents the posterior distribution over latent variables for an FA model.

    When creating this object, the user needs to specify the number of data points we will be working with.  The
    object will then be initialized with parameters for the mean for the posterior over the latents for each data
    point (the covariance for all data points is the same).

    In many applications we may not want the posterior distributions over all data points, so many of the methods
    of this object ask the user to specify the indices of data points relevant to the underlying computation.
    """

    def __init__(self, n_latent_vars: int, n_smps: int):
        """ Creates a new FAVariationalPosterior object.

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

    def sample(self, inds: torch.Tensor) -> torch.Tensor:
        """ Samples latent values from the posterior for given data points.

        Args:

            inds: The indices of latents to return samples for.  Of type long

        Returns:

            samples: samples[i,:] is the sample for inds[i]

        """

        n_data_pts = len(inds)

        return torch.randn(n_data_pts, self.m).mm(self.c.T) + self.mns[inds, :]


class PosteriorCollection():
    """ Contains posteriors over the parameters of an FA model.

    By holding all of the posteriors together, we gain convenience, when doing things such as moving the
    posteriors from one device to another or getting parameters.

    """

    def __init__(self, latent_post: FAVariationalPosterior, lm_post: OptionalDistribution = None,
                 mn_post: OptionalDistribution = None, psi_post: OptionalDistribution = None):
        """ Creates a new FAPosteriorCollection.

        Args:

            latent_post: The posterior over latent variables.

            lm_post: The posterior over the coefficients of the loading matrix.

            mn_post: The posterior over the mean coefficients of the mean vector.

            psi_post: The posterior over private variances.

        """

        self.latent_post = latent_post
        self.lm_post = lm_post
        self.mn_post = mn_post
        self.psi_post = psi_post

    def to(self, device: Union[str, torch.device]):
        """ Moves the collection the the specified device."""

        self.latent_post.to(device)

        if self.lm_post is not None:
            self.lm_post.to(device)
        if self.mn_post is not None:
            self.mn_post.to(device)
        if self.psi_post is not None:
            self.psi_post.to(device)

    def parameters(self) -> List[torch.nn.Parameter]:
        """ Returns all parameters in all distributions in the collection. """

        latent_post_parameters = list(self.latent_post.parameters())

        if self.lm_post is not None:
            lm_post_parameters = list(self.lm_post.parameters())
        if self.mn_post is not None:
            mn_post_parameters = list(self.mn_post.parameters())
        if self.psi_post is not None:
            psi_post_parameters = list(self.psi_post.parameters())

        return latent_post_parameters + lm_post_parameters + mn_post_parameters + psi_post_parameters


class PriorCollection():
    """ Contains conditional priors over the parameters of an FA model.

    By holding all of the priors together, we gain convenience, when doing things such as moving the
    priors from one device to another to getting the paraemeters of priors.

    """

    def __init__(self, lm_prior: OptionalDistribution, mn_prior: OptionalDistribution,
                 psi_prior: OptionalDistribution):
        """ Creates a new FAPriorCollection.

        Args:

            lm_prior: The conditional prior over the coefficients of the loading matrix.

            mn_prior: The conditional prior over the mean coefficients of the mean vector.

            psi_prior: The conditional prior over private variances.
        """

        self.lm_prior = lm_prior
        self.mn_prior = mn_prior
        self.psi_prior = psi_prior

    def to(self, device: Union[str, torch.device]):
        """ Moves the collection the the specified device."""

        if self.lm_prior is not None:
            self.lm_prior.to(device)
        if self.mn_prior is not None:
            self.mn_prior.to(device)
        if self.psi_prior is not None:
            self.psi_prior.to(device)

    def parameters(self) -> List[torch.nn.Parameter]:
        """ Returns all parameters present in all priors in the collection. """

        if self.lm_prior is not None:
            lm_parameters = list(self.lm_prior.parameters())
        else:
            lm_parameters = []

        if self.mn_prior is not None:
            mn_parameters = list(self.mn_prior.parameters())
        else:
            mn_parameters = []

        if self.psi_prior is not None:
            psi_parameters = list(self.psi_prior.parameters())
        else:
            psi_parameters = []

        return lm_parameters + mn_parameters + psi_parameters


def generate_simple_prior_collection(n_prop_vars: int, n_latent_vars: int, min_gaussian_std: float = .01,
                                     min_gamma_conc_vl: float = 1.0, min_gamma_rate_vl: float = .01,
                                     lm_mn_w_init_std: float = .01, lm_std_w_init_std: float = .01,
                                     mn_mn_w_init_std: float = .01, mn_std_w_init_std: float = .01,
                                     psi_conc_f_w_init_std: float = 1, psi_rate_f_w_init_std: float = .01,
                                     psi_conc_bias_mn: float = 10.0, psi_rate_bias_mn: float = 10.0) -> PriorCollection:
    """ Generates conditional priors where simple functions of properties generate distribution parameters.

    The conditional priors over the coefficients of the loading matrix and mean vectors will be Gaussian, with:

        1) Means which are a linear function of properties

        2) Standard deviations which are passed through linear function (of properties), the absolute value will
        be taken and then a fixed offset is added to prevent standard deviations from going below a certain bound.

    The conditional priors over the private variances will be Gamma distributions with:

        1) Concentration and rate parameters which are functions of the same form as (2) above.

    Args:
        n_prop_vars: The number of properties the distributions are conditioned on.

        n_latent_vars: The number of latent variables in the FA models.

        min_gaussian_std: The floor on the standard deviation for the distributions on the coefficients of the loading
        matrix and mean vector.

        min_gamma_conc_vl: The floor on values that the concentration parameter of the Gamma distributions can take
        on.

        min_gamma_rate_vl: The floor on values that rate parameter of the folded Gamma distributions can take on.

        lm_mn_w_init_std: When initializing the weights and biases of the linears functions predicting
        the mean of the distributions for the coefficients of the loading matrix, we initialize
        from centered normals, with standard deviations specified by this parameters.

        lm_std_w_init_std: Standard deviation for initializing weights and biases for linear function predicting the
        standard deviation of distributions over the loading matrix coefficients

        mn_mn_w_init_std: Standard deviation for initializing weights and biases for linear function predicting the
        mean of distributions over the coefficients of mean vectors

        mn_std_w_init_std: Standard deviation for initializing weights and biases for linear function predicting the
        standard deviation of distributions over the coefficents of mean vectors

        psi_conc_f_w_init_std: Standard deviation for initializing weights and biases for linear function predicting the
        concentration parameter of distributions over private variances

        psi_rate_f_w_init_std: Standard deviation for initializing weights and biases for linear function predicting the
        rate parameter of distributions over private variances

        psi_conc_bias_mn: The mean value for initializing the biases for the linear functions predicting the
        concentration parameters of distributions over private variances.

        psi_rate_bias_mn: The mean value for initializing the biases for the linear functions predicting the
        rate parameters of distributions over private variances.

    Returns:

        priors: The generated collection of priors.
    """

    # Generate prior for loadings matrix
    lm_mn_f = torch.nn.Linear(in_features=n_prop_vars, out_features=n_latent_vars, bias=True)
    lm_std_f = torch.nn.Sequential(torch.nn.Linear(in_features=n_prop_vars, out_features=n_latent_vars, bias=True),
                                   FixedOffsetAbs(o=min_gaussian_std))

    torch.nn.init.normal_(lm_mn_f.weight, mean=0.0, std=lm_mn_w_init_std)
    torch.nn.init.normal_(lm_mn_f.bias, mean=0.0, std=lm_mn_w_init_std)
    torch.nn.init.normal_(lm_std_f[0].weight, mean=0.0, std=lm_std_w_init_std)
    torch.nn.init.normal_(lm_std_f[0].bias, mean=0.0, std=lm_std_w_init_std)

    lm_prior = CondGaussianDistribution(mn_f=lm_mn_f, std_f=lm_std_f)

    mn_mn_f = torch.nn.Linear(in_features=n_prop_vars, out_features=1, bias=True)
    mn_std_f = torch.nn.Sequential(torch.nn.Linear(in_features=n_prop_vars, out_features=1, bias=True),
                                   FixedOffsetAbs(o=min_gaussian_std))

    torch.nn.init.normal_(mn_mn_f.weight, mean=0.0, std=mn_mn_w_init_std)
    torch.nn.init.normal_(mn_mn_f.bias, mean=0.0, std=mn_mn_w_init_std)
    torch.nn.init.normal_(mn_std_f[0].weight, mean=0.0, std=mn_std_w_init_std)
    torch.nn.init.normal_(mn_std_f[0].bias, mean=0.0, std=mn_std_w_init_std)

    mn_prior = CondGaussianDistribution(mn_f=mn_mn_f, std_f=mn_std_f)

    # Generate prior for private variances
    psi_conc_f = torch.nn.Sequential(torch.nn.Linear(in_features=n_prop_vars, out_features=1, bias=True),
                                     FixedOffsetAbs(o=min_gamma_conc_vl))
    psi_rate_f = torch.nn.Sequential(torch.nn.Linear(in_features=n_prop_vars, out_features=1, bias=True),
                                     FixedOffsetAbs(o=min_gamma_rate_vl))

    torch.nn.init.normal_(psi_conc_f[0].weight, mean=0.0, std=psi_conc_f_w_init_std)
    torch.nn.init.normal_(psi_conc_f[0].bias, mean=psi_conc_bias_mn, std=psi_conc_f_w_init_std)
    torch.nn.init.normal_(psi_rate_f[0].weight, mean=0.0, std=psi_rate_f_w_init_std)
    torch.nn.init.normal_(psi_rate_f[0].bias, mean=psi_rate_bias_mn, std=psi_rate_f_w_init_std)

    psi_prior = CondGammaDistribution(conc_f=psi_conc_f, rate_f=psi_rate_f)

    return PriorCollection(lm_prior=lm_prior, mn_prior=mn_prior, psi_prior=psi_prior)


def generate_basic_posteriors(n_obs_vars: Sequence[int], n_smps: Sequence[int], n_latent_vars: int,
                              lm_opts: OptionalDict = None, mn_opts: OptionalDict = None,
                              psi_opts: OptionalDict = None) -> List[PosteriorCollection]:
    """ Generates basic posteriors over model parameters and latents for a set of models.

    By basic posteriors, we mean the posteriors are not conditioned on properties but instead
    are represented by a product of distributions over each coefficient in the parameters, with
    the distribution for each coefficient being learned independently.

    We represent posterior distributions over the coefficients of the loading matrix and mean vector
    as Gaussians, and posterior distributions over private variances as Gamma distributions.

    The posterior over latents is represented as Gaussian distributions, with seperate means for each
    data point and a shared covariance matrix for all data points.

    Args:

        n_obs_vars: n_obs_vars[i] is the number of observed variables for model i

        n_smps: n_smps[i] is the total number of data points we will be fitting for model i

        n_latent_vars: The number of latent variables in the FA models.

        lm_opts: Dictionary of options to provide to MatrixGaussianProductDistribution when creating the
        posteriors over loading matrices.  See that object for available options.

        mn_opts: Dictionary of options to provide to MatrixGaussianProductDistribution when creating the
        posteriors over mean vectors.  See that object for available options.

        psi_opts: Dictionary of options to provide to GammaProductDistribution when creating the
        posteriors over private variances.  See that object for available options.

    Returns:

        post_collections: post_colletions[i] contains the posterior collections for subject i.

    """

    if lm_opts is None:
        lm_opts = dict()
    if mn_opts is None:
        mn_opts = dict()
    if psi_opts is None:
        psi_opts = dict()

    n_mdls = len(n_obs_vars)
    post_collections = [None]*n_mdls

    for i, n_i in enumerate(n_obs_vars):
        latent_post = FAVariationalPosterior(n_latent_vars=n_latent_vars, n_smps=n_smps[i])
        lm_post = MatrixGaussianProductDistribution(shape=[n_i, n_latent_vars], **lm_opts)
        mn_post = MatrixGaussianProductDistribution(shape=[n_i, 1], **mn_opts)
        #psi_post = FoldedNormalProductDistribution(n_vars=n_i, **psi_opts)
        psi_post = GammaProductDistribution(n_vars=n_i, **psi_opts)

        post_collections[i] = PosteriorCollection(latent_post=latent_post, lm_post=lm_post, mn_post=mn_post,
                                                  psi_post=psi_post)

    return post_collections


class VICollection():
    """ A collection of objects necessary for fitting one FA model with variational inference.

    This is a wrapper object that contains everything needed for fitting one FA model: it contains the data
    the model is fit to, the properties associated with each of the observed random variables in the model,
    the posterior distribution over model parameters, and the FA model object itself (if all the parameters of the
    model are fit probabilistically, the parameters within the FA object will be ignored).

    By placing all of these entities into a single object, certain things, like moving data and models between
    devices can be made more convenient.

     """

    def __init__(self, data: torch.Tensor, props: torch.Tensor, mdl: FAMdl, posteriors: PosteriorCollection):
        """ Creates a new VICollection object.

        Args:

            data: The data the FA model is fit to.  Of shape n_smps*n_obs_vars

            props: The properties associated with each of the observed random variables in the model.
            Of shape n_obs_vars*n_props

            mdl: The FA model object.  When creating this object, each parameter in the model for which posteriors
            will be fitted (instead of point estimates) should be set to None.

            posteriors: The collection of posteriors that will be fit over model parameters.

        """

        self.data = data
        self.props = props
        self.mdl = mdl
        self.posteriors = posteriors

    def to(self, device: Union[str, torch.device]):
        """ Moves everything in the collection to a device.

        Args:
            device: The device to move everything to.

        """

        self.data = self.data.to(device)
        self.props = self.props.to(device)
        self.mdl.to(device)
        self.posteriors.to(device)

    def parameters(self) -> List[torch.nn.Parameter]:
        """ Returns parameters in the FA model and all posteriors in the collection. """

        return list(self.mdl.parameters()) + self.posteriors.parameters()


class Fitter():
    """ Fits multiple FA models together, performing model synthesis, as well as estimating distributions over latents.

    Fitting can be performed in multiple ways.  The most generic way is that we learn a set of conditional priors over
    all model parameters as well as a set of posteriors over parameters for each individual model.

    Alternatively, for some parameters we may wish to learn point estimates alone. In this case, we do not learn a
    prior over these parameters nor posteriors.  Instead, for each model we simply learn a point estimate that
    maximizes the ELBO.  We indicate which parameters to learn posteriors over by setting their value to None in
    the FA model objects in each VI collection (see below).  If we set their values to some tensor, then the model
    will directly optimize these tensors, ignoring the posteriors and priors.  We expect we will treat parameters
    the same, learning them with point estimates or probabilistically, across models.

    In addition to the above, there may be times when we wish all models to share the same conditional posterior
    distributions over model parameters (this may be helpful for arriving at a good set of initial distributions
    before performing model synthesis where each model gets its own posterior distribution, and might also be
    useful if wanting to later perform amortized inference on model parameters).  In this case, the user can
    provide VI collections where the posteriors for these parameters are the same object.  In this case, the KL
    divergence between the posteriors and the prior can be trivially set to 0 by simply setting the prior equal
    to the posteriors, and fitting can be simplified by setting the "skip_kl" option for the parameters with
    shared posteriors to True when calling fit.

    """

    def __init__(self, vi_collections: Sequence[VICollection], priors: PriorCollection, min_psi: float = .0001):
        """ Creates a new Fitter object.

        Args:
            vi_collections: The VI collections to use for each model.

            priors: The collection of conditional priors to fit.

            min_psi: The minimum value that private variances can take on when sampling.  Sampled private variance
            values will be thresholded at this value.

        """
        self.vi_collections = vi_collections
        self.priors = priors
        self.min_psi = min_psi
        self.n_mdls = len(self.vi_collections)

    def fit(self, n_epochs: int, n_batches: int = 2, init_lr: float = .01, milestones: List[int] = None,
            gamma: float = .1, skip_lm_kl: bool = False, skip_mn_kl: bool = False,
            skip_psi_kl: bool = False, update_int: int = 10):
        """ Fits FA models together.

        Args:
            n_epochs: The number of epochs to run fitting for.

            n_batches: The number of batches to break the data up into during each epoch.

            init_lr: The initial learning rate to start optiimzation with

            milestones: A list of epochs at which to reduce the learning rate by a factor of gamma.  If not provided,
            the initial learning rate will be used the whole time.

            gamma: The factor to reduce the learning rate by at each milestone in milestones

            skip_lm_kl: If true, kl divergences between posteriors and the prior for loading matrices are not calculated.
            This can be safely set to true when all posteriors are the same shared conditional posterior (in which
            minimizing the KL divergence can be achieved trivially by setting the prior also equal to the shared
            conditional posterior).

            skip_mn_kl: If true, kl divergences between posteriors and the prior for mean vectors are not calculated.

            skip_psi_kl: If true, kl divergences between posteriors and the prior for the private variances are not
            calculated.

            update_int: The number of epochs after which we provide the user with a status update

        Returns:

            log: A dictionary with the following keys:
                obj: A numpy array of shape n_epochs.  obj[e_i] is the objective value after epoch e_i

                nell: A numpy array of shape n_epochs*n_models.  obj[e_i,m_i] is the negative expected log likelihood,
                averaged across batches after epoch e_i for model m_i

                latent_kl: A numpy array of shape n_epochs*n_models.  latent_kl[e_i,m_i] is the kl divergence
                between the prior distribution over latents and the posterior averaged across batches after epoch e_i
                for model m_i

                lm_kl: A numpy array of shape n_epochs*n_models.  lm_kl[e_i,m_i] is the kl divergence
                between the prior distribution over loading matrices and the posterior averaged across batches after
                epoch e_i for model m_i

                mn_kl: A numpy array of shape n_epochs*n_models.  mn_kl[e_i,m_i] is the kl divergence
                between the prior distribution over mean vectors and the posterior averaged across batches after
                epoch e_i for model m_i

                psi_kl: A numpy array of shape n_epochs*n_models.  psi_kl[e_i,m_i] is the kl divergence
                between the prior distribution over private variances and the posterior averaged across batches after
                epoch e_i for model m_i
        """

        # If milestones are not provided, we set things up to use the initial learning rate the whole time
        if milestones is None:
            milestones = [n_epochs + 1]

        # Determine if there are any parameters we are estimating with point estimates
        lm_point_estimate = self.vi_collections[0].mdl.lm is not None
        mn_point_estimate = self.vi_collections[0].mdl.mn is not None
        psi_point_estimate = self.vi_collections[0].mdl.psi is not None

        # Gather all parameters we will be optimizing - note that this code assumes the user has not be "wasteful" in
        # creating parameters that will not be optimized (e.g., by creating posteriors and priors for parameters we are
        # estimating with point estimates). However, if the user has been "wasteful" the code below will still produce
        # the correct computations; we will just waste space carrying around and optimizing tensors which do not factor
        # into the objective we are optimizing.

        params = self.priors.parameters() + list(itertools.chain(*[coll.parameters() for coll in self.vi_collections]))

        # Make sure we have no duplicate parameters
        params = list(set(params))

        optimizer = torch.optim.Adam(params=params, lr=init_lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=gamma)

        # Determine the total number of data points we have for fitting each model
        model_n_data_pts = [coll.data.shape[0] for coll in self.vi_collections]

        # Set things up for logging
        obj_log = np.zeros(n_epochs)
        nell_log = np.zeros([n_epochs, self.n_mdls])
        latent_kl_log = np.zeros([n_epochs, self.n_mdls])
        lm_kl_log = np.zeros([n_epochs, self.n_mdls])
        mn_kl_log = np.zeros([n_epochs, self.n_mdls])
        psi_kl_log = np.zeros([n_epochs, self.n_mdls])

        # Optimization loop
        for e_i in range(n_epochs):

            # Determine which samples we use for each batch for each subject
            batch_smp_inds = self.generate_batch_smp_inds(n_batches)

            # Create variables for logging results across batches
            batch_nell_log = np.zeros([n_batches, self.n_mdls])
            batch_latent_kl_log = np.zeros([n_batches, self.n_mdls])
            batch_lm_kl_log = np.zeros([n_batches, self.n_mdls])
            batch_mn_kl_log = np.zeros([n_batches, self.n_mdls])
            batch_psi_kl_log = np.zeros([n_batches, self.n_mdls])

            for b_i in range(n_batches):

                obj = 0 # Keeps track of objective, summed across models
                optimizer.zero_grad()

                for m_i in range(self.n_mdls):
                    mdl_posteriors = self.vi_collections[m_i].posteriors
                    mdl_props = self.vi_collections[m_i].props
                    fa_mdl = self.vi_collections[m_i].mdl

                    batch_inds = batch_smp_inds[m_i][b_i]
                    data_b_i = self.vi_collections[m_i].data[batch_inds, :]
                    n_batch_data_pts = data_b_i.shape[0]

                    # Sample our posteriors
                    latents_smp = mdl_posteriors.latent_post.sample(inds=batch_inds)

                    if not lm_point_estimate:
                        # We produce samples in both compact and standard form.  Compact form is used for
                        # computing KL divergences; standard form is used when computing likelihoods
                        lm_compact_smp, lm_standard_smp = self._sample_posterior(post=mdl_posteriors.lm_post,
                                                                                 props=mdl_props)
                    else:
                        lm_standard_smp = None  # Passing in None to the appropriate functions will signal we use the
                                                # parameter stored in the FA model object

                    if not mn_point_estimate:
                        mn_compact_smp, mn_standard_smp = self._sample_posterior(post=mdl_posteriors.mn_post,
                                                                                 props=mdl_props)
                        mn_standard_smp = mn_standard_smp.squeeze()
                    else:
                        mn_standard_smp = None

                    if not psi_point_estimate:
                        psi_compact_smp, psi_standard_smp = self._sample_posterior(post=mdl_posteriors.psi_post,
                                                                                   props=mdl_props)
                        psi_standard_smp = psi_standard_smp.squeeze()

                        # Enforce floor on sampled private variances
                        enforce_floor(psi_compact_smp, self.min_psi)
                        enforce_floor(psi_standard_smp, self.min_psi)

                    else:
                        psi_standard_smp = None

                    # Compute expected log-likelihood
                    corr_f = float(model_n_data_pts[m_i])/n_batch_data_pts
                    nell = -corr_f*torch.sum(fa_mdl.cond_log_prob(z=latents_smp, x=data_b_i, lm=lm_standard_smp,
                                                                  mn=mn_standard_smp, psi=psi_standard_smp))

                    # Compute KL divergences
                    latent_kl = corr_f*mdl_posteriors.latent_post.kl_btw_standard_normal(inds=batch_inds)

                    if (not lm_point_estimate) and (not skip_lm_kl):
                        lm_kl = torch.sum(mdl_posteriors.lm_post.kl(d_2=self.priors.lm_prior, x=mdl_props, smp=lm_compact_smp))
                    else:
                        lm_kl = 0
                    if (not mn_point_estimate) and (not skip_mn_kl):
                        mn_kl = torch.sum(mdl_posteriors.mn_post.kl(d_2=self.priors.mn_prior, x=mdl_props, smp=mn_compact_smp))
                    else:
                        mn_kl = 0
                    if (not psi_point_estimate) and (not skip_psi_kl):
                        psi_kl = torch.sum(mdl_posteriors.psi_post.kl(d_2=self.priors.psi_prior, x=mdl_props, smp=psi_compact_smp))
                    else:
                        psi_kl = 0

                    # Calculate gradients for this batch
                    mdl_obj = nell + latent_kl + lm_kl + mn_kl + psi_kl
                    mdl_obj.backward()
                    obj += get_scalar_vl(mdl_obj)

                    # Log progress
                    batch_nell_log[b_i, m_i] = get_scalar_vl(nell)
                    batch_latent_kl_log[b_i, m_i] = get_scalar_vl(latent_kl)
                    batch_lm_kl_log[b_i, m_i] = get_scalar_vl(lm_kl)
                    batch_mn_kl_log[b_i, m_i] = get_scalar_vl(mn_kl)
                    batch_psi_kl_log[b_i, m_i] = get_scalar_vl(psi_kl)

                optimizer.step()

            # Enforce floor on private variances if estimating them with point estimates
            if psi_point_estimate:
                for coll in self.vi_collections:
                    enforce_floor(coll.mdl.psi, self.min_psi)

            # Update logs
            obj_log[e_i] = obj
            nell_log[e_i, :] = np.mean(batch_nell_log, axis=0)
            latent_kl_log[e_i, :] = np.mean(batch_latent_kl_log, axis=0)
            lm_kl_log[e_i, :] = np.mean(batch_lm_kl_log, axis=0)
            mn_kl_log[e_i, :] = np.mean(batch_mn_kl_log, axis=0)
            psi_kl_log[e_i, :] = np.mean(batch_psi_kl_log, axis=0)

            if e_i % update_int == 0:
                self._print_status_update(epoch_i=e_i, obj_v=obj_log[e_i], nell_v=nell_log[e_i, :],
                                          latent_kl_v=latent_kl_log[e_i, :], lm_kl_v=lm_kl_log[e_i, :],
                                          mn_kl_v=mn_kl_log[e_i, :], psi_kl_v=psi_kl_log[e_i, :],
                                          lr=get_lr(optimizer))

            scheduler.step()

        # Generate final log structure
        log = {'obj': obj_log, 'nell': nell_log, 'latent_kl': latent_kl_log, 'lm_kl': lm_kl_log,
               'mn_kl': mn_kl_log, 'psi_kl': psi_kl_log}

        return log

    def generate_batch_smp_inds(self, n_batches: int) -> List[List[np.ndarray]]:
        """ Generates indices of random mini-batches of samples for each model.

        Args:
            n_batches: The number of batches to break the data up for each subject into.

        Returns:
            batch_smp_inds: batch_smp_inds[i][j] is the sample indices for the j^th batch for the i^th model

        """

        n_smps = [self.vi_collections[i].data.shape[0] for i in range(self.n_mdls)]

        for i, n_s in enumerate(n_smps):
            if n_s < n_batches:
                raise(ValueError('Data for model ' + str(i) + ' has only ' + str(n_s) + ' samples, while '
                      + str(n_batches) + ' batches requested.'))

        batch_sizes = [int(np.floor(float(n_s)/n_batches)) for n_s in n_smps]

        batch_smp_inds = [None]*self.n_mdls
        for i in range(self.n_mdls):
            subject_batch_smp_inds = [None]*n_batches
            perm_inds = np.random.permutation(n_smps[i])
            start_smp_ind = 0
            for b_i in range(n_batches):
                if b_i == (n_batches - 1):
                    end_smp_ind = n_smps[i]
                else:
                    end_smp_ind = start_smp_ind+batch_sizes[i]
                subject_batch_smp_inds[b_i] = torch.tensor(perm_inds[start_smp_ind:end_smp_ind], dtype=torch.long)
                start_smp_ind = end_smp_ind
            batch_smp_inds[i] = subject_batch_smp_inds

        return batch_smp_inds

    @staticmethod
    def plot_log(log: dict):
        """ Generates a figure showing logged values. """

        plt.figure()

        ax = plt.subplot(3, 2, 1)
        ax.plot(log['obj'])
        plt.xlabel('Epoch')
        plt.ylabel('Objective')

        ax = plt.subplot(3, 2, 2)
        ax.plot(log['nell'])
        plt.xlabel('Epoch')
        plt.ylabel('NELL')

        ax = plt.subplot(3, 2, 3)
        ax.plot(log['latent_kl'])
        plt.xlabel('Epoch')
        plt.ylabel('Latent KL')

        ax = plt.subplot(3, 2, 4)
        ax.plot(log['lm_kl'])
        plt.xlabel('Epoch')
        plt.ylabel('LM KL')

        ax = plt.subplot(3, 2, 5)
        ax.plot(log['mn_kl'])
        plt.xlabel('Epoch')
        plt.ylabel('Mn KL')

        ax = plt.subplot(3, 2, 6)
        ax.plot(log['psi_kl'])
        plt.xlabel('Epoch')
        plt.ylabel('Psi KL')

    @staticmethod
    def _print_status_update(epoch_i, obj_v, nell_v, latent_kl_v, lm_kl_v, mn_kl_v, psi_kl_v, lr):
        """ Prints a formatted status update to the screen. """
        print('')
        print('=========== EPOCH ' + str(epoch_i) + ' COMPLETE ===========')
        print('Obj: {:.2e}'.format(obj_v))
        print('----------------------------------------')
        print('NELL: ' + list_to_str(nell_v))
        print('Latent KL: ' + list_to_str(latent_kl_v))
        print('LM KL: ' + list_to_str(lm_kl_v))
        print('Mn KL: ' + list_to_str(mn_kl_v))
        print('Psi KL: ' + list_to_str(psi_kl_v))
        print('----------------------------------------')
        print('LR: ' + str(lr))

    @staticmethod
    def _sample_posterior(post: CondVAEDistribution, props: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        compact_smp = post.sample(props)
        standard_smp = post.form_standard_sample(compact_smp)
        return compact_smp, standard_smp


