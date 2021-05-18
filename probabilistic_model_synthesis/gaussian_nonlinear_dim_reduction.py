""" Tools for performing model synthesis over non-linear dimensionality reduction models with Gaussian noise. """

import copy
import glob
import itertools
import math
import pathlib
import time
from typing import List, Optional, Tuple, Sequence, Union

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import numpy.linalg
import torch
import torch.optim


from janelia_core.math.basic_functions import list_grid_pts
from janelia_core.ml.extra_torch_modules import ConstantRealFcn
from janelia_core.ml.extra_torch_modules import FixedOffsetAbs
from janelia_core.ml.extra_torch_modules import SumOfTiledHyperCubeBasisFcns
from janelia_core.ml.torch_distributions import CondGaussianDistribution
from janelia_core.ml.torch_distributions import CondMatrixHypercubePrior
from janelia_core.ml.torch_distributions import CondVAEDistribution
from janelia_core.ml.torch_distributions import MatrixGaussianProductDistribution
from janelia_core.ml.utils import summarize_memory_stats
from janelia_core.visualization.matrix_visualization import cmp_n_mats
from probabilistic_model_synthesis.distributions import CondGammaDistribution
from probabilistic_model_synthesis.distributions import GammaProductDistribution
from probabilistic_model_synthesis.distributions import SampleLatentsGaussianVariationalPosterior
from probabilistic_model_synthesis.utilities import enforce_floor
from probabilistic_model_synthesis.utilities import get_lr
from probabilistic_model_synthesis.utilities import get_scalar_vl
from probabilistic_model_synthesis.utilities import list_to_str


# Define type aliases
OptionalDevices = Optional[Sequence[torch.device]]
OptionalDict = Optional[dict]
OptionalDistribution = Optional[CondVAEDistribution]
OptionalTensor = Optional[torch.Tensor]
StrOrPath = Union[pathlib.Path, str]


def align_intermediate_spaces(mn0: np.ndarray, lm0: np.ndarray, s0: np.ndarray,
                              mn1: np.ndarray, lm1: np.ndarray, s1: np.ndarray,
                              int_z0: Optional[np.ndarray] = None,
                              int_z1: Optional[np.ndarray] = None,
                              align_by_params: bool = True) -> Tuple[np.ndarray]:
    """ Aligns latent representations in intermediate spaces between models as well as loading matrices and mean vectors

    The intermediate latent space of a model is the range of the m module which maps between the low-d representation
    of latents and their (usually) higher-d representation.  There is a linear mapping between latents in the
    intermediate space and mean observed values.  This linear mapping is given by the loading matrix and mean vector.

    This function will align the intermediate latent space of model 1 to that of model 0.  In doing so, it will
    transform the loading matrix and mean vector for model 1 to be appropriate for this alignment, and it will
    also (optionally) transform a set of latent points in the original intermediate space to this new space.

    The alignment can be done in two ways.  In the first way, the alignment is based on comparing the loading
    matrices and mean vectors between the two models.  In the second way, the alignment is based on trying to align
    individual latent points between the two spaces themselves.

    Args:

        mn0: The mean vector for model 0

        lm0: The loading matrix  for model 0

        s0: The scales for model 0

        mn1: The mean vector for model 1

        lm1: The loading matrix for model 1

        s1: The scales matrix for model 1

        int_z0: The latents in the intermediate space for model 0. Of shape n_smps*n_intermediate_dims

        int_z1: The latents in the intermediate space for model 1. Of shape n_smps*n_intermediate_dims

        align_by_params: If true, alignment is performed based on aligning the parameters of the models.  If false,
        alignment is performed by aligned the latent points themselves.

    Returns:

        mn1: The new mean for model 1, appropriate for the aligned intermediate latent space

        lm1: The new loading matrix for model 1, appropriate for the aligned intermediate latent space

        w: The transformation matrix used to align latents and parameters of model 1 to model 0.

        int_z1: The new intermediate latents for model 1, transformed into the aligned space.  This output is
        only provided if int_z1 is provided in the input

    """

    # Make local copies of input variables
    mn0 = copy.deepcopy(mn0)
    lm0 = copy.deepcopy(lm0)
    s0 = copy.deepcopy(s0)
    mn1 = copy.deepcopy(mn1)
    lm1 = copy.deepcopy(lm1)
    s1 = copy.deepcopy(s1)
    int_z0 = copy.deepcopy(int_z0)
    int_z1 = copy.deepcopy(int_z1)

    # Perform alignment of intermediate latent spaces
    mn0 = mn0 * np.abs(s1)
    lm0 = lm0 * np.abs(np.expand_dims(s0, 1))
    mn1 = mn1 * np.abs(s1)
    lm1 = lm1 * np.abs(np.expand_dims(s0, 1))

    m0_conc = np.concatenate([lm0, np.expand_dims(mn0, 1)], axis=1)
    m1_conc = np.concatenate([lm1, np.expand_dims(mn1, 1)], axis=1)

    if align_by_params:
        u_0, s_0, v_0 = numpy.linalg.svd(m0_conc, full_matrices=False)
        w = np.matmul(v_0.transpose(), np.matmul(np.diag(1/s_0), np.matmul(u_0.transpose(), m1_conc)))

    else:

        int_z0_ones = np.concatenate([int_z0, np.ones([int_z1.shape[0], 1])], axis=1)
        int_z1_ones = np.concatenate([int_z1, np.ones([int_z1.shape[0], 1])], axis=1)
        w = np.linalg.lstsq(int_z1_ones, int_z0_ones, rcond=None)
        w = w[0].transpose()

    m1_conc = np.matmul(m1_conc, np.linalg.inv(w))
    lm1 = m1_conc[:, 0:-1]
    mn1 = m1_conc[:, -1]

    if int_z1 is not None:
        int_z1 = np.matmul(np.concatenate([int_z1, np.ones([int_z1.shape[0], 1])], axis=1), w.transpose())
        int_z1 = int_z1[:, 0:-1]

    if int_z1 is not None:
        return mn1, lm1, w, int_z1
    else:
        return mn1, lm1, w


def compare_mean_and_lm_dists(lm_0_prior: CondVAEDistribution, mn_0_prior: CondVAEDistribution,
                              s_0_prior: CondVAEDistribution, lm_1_prior: CondVAEDistribution,
                              mn_1_prior: CondVAEDistribution, s_1_prior: CondVAEDistribution,
                              dim_0_range: Sequence, dim_1_range: Sequence, n_pts_per_dim: Sequence):
    """ Visualizes two sets of conditional distributions over means and loading matrices, aligned to one another.

    The second set of distributions will be aligned to the first.

    Note: The function assumes 2-d conditioning variables.

    Args:

        lm_0_prior: The conditional prior over loading matrices for the first model

        mn_0_prior: The conditional prior over means for the first model

        s_0_prior: The conditional prior over scales for the first model.

        lm_1_prior: The conditional prior over loading matrices for the second model

        mn_1_prior: The conditional prior over means for the second model

        s_1_prior: The conditional prior over scales for the second model.

        dim_0_range: The range of values of the form [start, stop] for the first dimension of conditioning values we
        view means and standard deviations over.

        dim_1_range: The range of values of the form [start, stop] for the second dimension of conditioning values we
        view means and standard deviations over.

        n_pts_per_dim: The number of points per dimension of the form [n_dim_1_pts, n_dim_2_pts] we view conditioning
        values on for each dimension.

    """

    pts, dim_pts = list_grid_pts(grid_limits=np.asarray([dim_0_range, dim_1_range]), n_pts_per_dim=n_pts_per_dim)
    pts = torch.tensor(pts)

    lm0_mn = lm_0_prior(pts).detach().numpy()
    mn0_mn = mn_0_prior(pts).detach().numpy().squeeze()
    s0_mn = s_0_prior(pts).detach().numpy().squeeze()
    lm1_mn = lm_1_prior(pts).detach().numpy()
    mn1_mn = mn_1_prior(pts).detach().numpy().squeeze()
    s1_mn = s_1_prior(pts).detach().numpy().squeeze()

    # Determine alignment between intermediate latent spaces and align means
    mn1_mn_al, lm1_mn_al, w = align_intermediate_spaces(lm0=lm0_mn, mn0=mn0_mn, s0=s0_mn,
                                                        lm1=lm1_mn, mn1=mn1_mn, s1=s1_mn)

    # Now get aligned standard deviations
    if 'dists' in dir(lm_1_prior):
        lm1_std = np.concatenate([d.std_f(pts).detach().numpy() for d in lm_1_prior.dists], axis=1)
    else:
        lm1_std = lm_1_prior.std_f(pts).detach().numpy()

    mn1_std = mn_1_prior.std_f(pts).detach().numpy()

    w_inv = np.linalg.inv(w)
    std1 = np.concatenate([lm1_std, mn1_std], axis=1)
    std1_al = np.zeros(std1.shape)
    for i, std_i in enumerate(std1):
        std1_al[i, :] = np.diag(np.matmul(w_inv.transpose(), np.matmul(np.diag(std_i), w_inv)))

    lm1_std_al = std1_al[:, 0:-1]
    mn1_std_al = std1_al[:, -1]

    # Now get standard deviations of the model we align to
    if 'dists' in dir(lm_0_prior):
        lm0_std = np.concatenate([d.std_f(pts).detach().numpy() for d in lm_0_prior.dists], axis=1)
    else:
        lm0_std = lm_0_prior.std_f(pts).detach().numpy()

    mn0_std = mn_0_prior.std_f(pts).detach().numpy()

    # Generate the figure
    plt.figure()
    n_int_latent_dims = lm1_mn_al.shape[1]
    n_rows = 1 + n_int_latent_dims

    def _plot_image(n_rows, n_cols, im_i, im_pts, title_str):
        ax = plt.subplot(n_rows, n_cols, im_i)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        im = ax.imshow(im_pts.reshape(n_pts_per_dim))
        plt.colorbar(im, cax=cax, orientation='vertical')
        ax.set_title(title_str)

    # Generate panels for mean
    _plot_image(n_rows, 4, 1, mn0_mn, 'Mean 0 Mn')
    _plot_image(n_rows, 4, 2, mn1_mn_al, 'Mean 1 Mn')
    _plot_image(n_rows, 4, 3, mn0_std, 'Mean 0 Std')
    _plot_image(n_rows, 4, 4, mn1_std_al, 'Mean 1 Std')

    # Now generate images for modes
    cnt = 4
    for i in range(n_int_latent_dims):
        # Plot true mapping for loading matrix means for this column of the loading matrix
        cnt += 1
        _plot_image(n_rows, 4, cnt, lm0_mn[:, i], 'LM 0:' + str(i) + ' Mn')
        cnt += 1
        _plot_image(n_rows, 4, cnt, lm1_mn_al[:, i], 'LM 1:' + str(i) + ' Mn')
        cnt += 1
        _plot_image(n_rows, 4, cnt, lm0_std[:, i], 'LM 0:' + str(i) + ' Std')
        cnt += 1
        _plot_image(n_rows, 4, cnt, lm1_std_al[:, i], 'LM 1:' + str(i) + ' Std')


class GNLDRMdl(torch.nn.Module):
    """ A nonlinear dimensionality reduction model with Gaussian noise on observed variables.

    A Gaussian nonlinear dimensionality reduction (GNLDR) model can be described by the following generative process.
    First, mean values of observed variables are generated according to:

        z_t ~ N(0,I), z_t \in R^p
        l_t = m(z_t) l_t \in R^q
        mn_t = abs(s)*(\Lambda l_t + \mu), Lambda \in R^{n \times q}, mu \in R^n, s \in R^b, where * represents an
                                      element-wise product
        x_t = mn_t + ep_t, ep_t ~ N(0, \Psi) for some diagonal PSD matrix Psi

    where m() is a neural network which lifts latent z_t into a (typically) higher dimensional space (the space of
    l_t) for which there is a linear mapping (parameterized by \Lambda, \mu and s) between l_t and mean values of
    observed variables.

    Here the vector s applies a scaling to the mean of each observed variable.  While it is algebraically redundant
    in the above model formulation, it becomes important when fitting models across experiments, where the scale of
    values observed variables take on may change across experiments, and we do want these changes to be reflected in the
    \Lambda and \mu values we learn for each experiment, so we instead absorb them in s.

    This object by default will use it's own internal parameters, but it's various functions allow a user
    to provide their own parameter values, making it easy to use these objects when fitting models and
    wanting point estimates for some parameters (which would be represented in the internal parameters of this
    object) while treating other parameters as random variables.
    """

    def __init__(self, n_latent_vars: int, m: torch.nn.Module, lm: OptionalTensor = None, mn: OptionalTensor = None,
                 psi: OptionalTensor = None, s: OptionalTensor = None):
        """ Creates a new GNLDRMdl Object.

        Note: when creating this object the user can supply values for the loading matrix, mean vector, private
        variance or s vector or optionally set any of these to None.  If they are set to None, no parameters will be
        created for them in the model.  In this case, it will be expected that values for them will be provided when
        calling any function (e.g., sample) that need these values.  The reason for this is that model
        objects may be used in frameworks which treat some/all of the parameters of a model as random variables, and we
        will want to provide sampled values of these parameters when working with the models, so there is no point in
        taking up memory representing them within the model object itself.

        Args:

            n_latent_vars: The number of latent variables in the model (the dimensionality of the low-d space)

            m: The mapping from the low-d latent space to the higher-d latent space.

            lm: The loading matrix of shape n_obs_variables*n_latent_dims

            mn: The mean vector of shape n_obs_variables

            psi: The private variances of shape n_obs_variables

            s: The vector of scales of shape n_obs_variables.

        """
        super().__init__()

        self.n_latent_vars = n_latent_vars
        self.m = m

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

        if s is not None:
            self.s = torch.nn.Parameter(s)
        else:
            self.s = None

        self.register_buffer('log_2_pi', torch.log(torch.tensor(2*math.pi)))

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

        mns = self.cond_mean(z=z, lm=lm, mn=mn, s=s)

        ll = -.5*torch.sum(((x - mns)**2)/psi, dim=1)
        ll -= .5*torch.sum(torch.log(psi))
        ll -= .5*n_obs_vars*self.log_2_pi

        return ll

    def cond_mean(self, z: torch.Tensor, lm: OptionalTensor = None, mn: OptionalTensor = None,
                  s: OptionalTensor = None) -> torch.Tensor:
        """ Computes the mean of observations given latents.

           Args:
               z: Latents of shape n_smps*n_latent_dims

               lm: If provided, uses this in place of the loading matrix parameter of the model.

               mn: If provided, uses this in place of the mean parameter of the model.

               s: If provided, use this in place of the scales parameters of the model.

           Returns:

                mn: The conditional mean for each sample. Of shape n_smps*n_obs_variables.

           """

        if lm is None:
            lm = self.lm
        if mn is None:
            mn = self.mn
        if s is None:
            s = self.s

        mns = torch.abs(s)*(torch.matmul(self.m(z), lm.T) + mn)

        return mns

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

        x_mn = self.cond_mean(z=z, lm=lm, mn=mn, s=s)
        x_noise = torch.randn(n_smps, n_obs_vars)*torch.sqrt(psi)
        x = x_mn + x_noise

        return z, x

    @staticmethod
    def compare_models(m0, m1):
        """ Visually compares two models.

        This function will find a linear transformation that best maps the weights of m1 to those m0 when displaying
        the weights for m1. Note that a model depending on the form of non-linearity in a model, the weights
        themselves may not be identifiable, even up to a learn transformation, so users should interpret differences
        in weights between models with care.

        Args:
            m0: The fist model

            m1: The second model
        """

        ROW_SPAN = 24  # Number of rows in the gridspec
        COL_SPAN = 24  # Number of columns in the gridspec

        # Copy models to local variable so changes to weights don't affect the passed object
        m0 = copy.deepcopy(m0)
        m1 = copy.deepcopy(m1)

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

        # Align the intermediate latent space of m1 to that of m0
        s0 = m0.s.detach().numpy()
        lm0 = m0.lm.detach().numpy()
        mn0 = m0.mn.detach().numpy()

        s1 = m1.s.detach().numpy()
        lm1 = m1.lm.detach().numpy()*np.abs(np.expand_dims(s1, 1))
        mn1 = m1.mn.detach().numpy()*np.abs(s1)

        mn1, lm1, _ = align_intermediate_spaces(mn0=mn0, lm0=lm0, s0=s0, mn1=mn1, lm1=lm1, s1=s1)

        # The transformed means and loading matrices have scales applied, so we apply them to model 0 as well
        lm0 = m0.lm.detach().numpy()*np.abs(np.expand_dims(s0, 1))
        mn0 = m0.mn.detach().numpy()*np.abs(s0)

        # Make plots of scalar variables
        _make_subplot([0, 0], 6, 6, mn0, mn1, 'Mean')
        _make_subplot([8, 0], 6, 6, m0.psi.cpu().detach().numpy(), m1.psi.cpu().detach().numpy(), 'Psi')
        _make_subplot([16, 0], 6, 6, np.abs(m0.s.cpu().detach().numpy()), np.abs(m1.s.cpu().detach().numpy()), 'S')

        # Make plots of loading matrices
        lm_diff = lm0 - lm1

        w1_grid_info = {'grid_spec': grid_spec}
        w1_cell_info = list()
        w1_cell_info.append({'loc': [0, 10], 'rowspan': 24, 'colspan': 3})
        w1_cell_info.append({'loc': [0, 15], 'rowspan': 24, 'colspan': 3})
        w1_cell_info.append({'loc': [0, 20], 'rowspan': 24, 'colspan': 3})
        w1_grid_info['cell_info'] = w1_cell_info
        cmp_n_mats([lm0, lm1, lm_diff], show_colorbars=True, titles=['LM 0', 'LM 1', 'LM 0 - LM 1'],
                   grid_info=w1_grid_info)


class Fitter():
    """ Fits multiple GNLDR models together, performing model synthesis, as well as estimating distributions over
    latents.

    Fitting can be performed in multiple ways.  The most generic way is that we learn a set of conditional priors over
    all model parameters as well as a set of posteriors over parameters for each individual model.

    Alternatively, for some parameters we may wish to learn point estimates alone. In this case, we do not learn a
    prior over these parameters nor posteriors.  Instead, for each model we simply learn a point estimate that
    maximizes the ELBO.  We indicate which parameters to learn posteriors over by setting their value to None in
    the model objects in each VI collection (see below).  If we set their values to some tensor, then the model
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

    def __init__(self, vi_collections: Sequence['VICollection'], priors: 'PriorCollection',
                 devices: OptionalDevices = None, min_psi: float = .0001):
        """ Creates a new Fitter object.

        Args:
            vi_collections: The VI collections to use for each model.

            priors: The collection of conditional priors to fit.

            min_psi: The minimum value that private variances can take on when sampling.  Sampled private variance
            values will be thresholded at this value.

        """
        self.vi_collections = vi_collections
        self.vi_collection_devices = None  # Keep track of devices we move each vi collection to
        self.priors = priors
        self.min_psi = min_psi
        self.n_mdls = len(self.vi_collections)

        if devices is None:
            devices = [torch.device('cpu')]
        self.devices = devices

    def distribute(self, distribute_data: bool = True, devices: OptionalDevices = None):
        """ Distributes priors and vi collections across devices.

        Args:

            distribute_data: True if data should be distributed to devices.  Setting to false, will leave the
            data on cpu. In this case, when fitting only mini-batches of data will be sent to GPU. This can
            help with memory consumption but increases fitting time.

            devices: Devices to distribute across.  If provided, the devices for the fitter object will be updated
            to these. If not provided, the devices already in the objects devices attribute will be used.
        """

        if devices is not None:
            self.devices = devices

        # Priors always go on first device
        self.priors.to(device=self.devices[0])

        n_devices = len(self.devices)
        n_vi_collections = len(self.vi_collections)
        vi_collection_devices = [None]*n_vi_collections
        for c_i in range(n_vi_collections):
            d_i = c_i % n_devices
            self.vi_collections[c_i].to(device=self.devices[d_i], move_data=distribute_data)
            vi_collection_devices[c_i] = self.devices[d_i]

        self.vi_collection_devices = vi_collection_devices

    def fit(self, n_epochs: int, n_batches: int = 2, init_lr: float = .01, milestones: List[int] = None,
            gamma: float = .1, skip_lm_kl: bool = False, skip_mn_kl: bool = False,
            skip_psi_kl: bool = False, skip_s_kl: bool = False, update_int: int = 10, cp_epochs: Sequence[int] = None,
            cp_save_folder: StrOrPath = None, cp_save_str: str = '', optimize_only_latents: bool = False):
        """ Fits GNLDR models together.

        Args:
            n_epochs: The number of epochs to run fitting for.

            n_batches: The number of batches to break the data up into during each epoch.

            init_lr: The initial learning rate to start optimzation with

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

            skip_s_kl: If true, kl divergences between posteriors and the prior for the scales are not calculated.

            update_int: The number of epochs after which we provide the user with a status update.  If None,
            no updates will be printed.

            cp_epochs: A sequence of epochs after which check point should be created.

            cp_save_folder: A folder where check points should be saved

            cp_save_str: A string to add to the name of saved check point files (see save_checkpoint() for more details).

            optimize_only_latents: If true, only the parameters of the distributions over latents are are optimized.
            This is useful when wanting to hold the distributions over model parameters fixed but infer latents for
            new data points.

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

                s_kl: A numpy array of shape n_epochs*n_models.  s_kl[e_i,m_i] is the kl divergence
                between the prior distribution over scales and the posterior averaged across batches after
                epoch e_i for model m_i

        """

        if cp_epochs is None:
            cp_epochs = []

        # Make sure we have distributed priors and vi collections across devices
        if self.vi_collection_devices is None:
            raise(RuntimeError('Distribute must be called before fit.'))

        # If milestones are not provided, we set things up to use the initial learning rate the whole time
        if milestones is None:
            milestones = [n_epochs + 1]

        # See if we are estimating psi with a point estimate, so we can enforce floors on these estimates if we are
        psi_point_estimate = self.vi_collections[0].mdl.psi is not None

        # Gather all parameters we will be optimizing - note that this code assumes the user has not been "wasteful" in
        # creating parameters that will not be optimized (e.g., by creating posteriors and priors for parameters we are
        # estimating with point estimates). However, if the user has been "wasteful" the code below will still produce
        # the correct computations; we will just waste space carrying around and optimizing tensors which do not factor
        # into the objective we are optimizing.

        if not optimize_only_latents:
            params = self.priors.parameters() + list(itertools.chain(*[coll.parameters()
                                                                       for coll in self.vi_collections]))
        else:
            params = list(itertools.chain(*[coll.posteriors.latent_post.parameters() for coll in self.vi_collections]))
            skip_mn_kl = True
            skip_lm_kl = True
            skip_psi_kl = True
            skip_s_kl = True

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
        s_kl_log = np.zeros([n_epochs, self.n_mdls])

        # Optimization loop
        t_0 = time.time()
        for e_i in range(n_epochs):

            # Determine which samples we use for each batch for each subject
            batch_smp_inds = self.generate_batch_smp_inds(n_batches)

            # Create variables for logging results across batches
            batch_obj = np.zeros(n_batches)
            batch_nell_log = np.zeros([n_batches, self.n_mdls])
            batch_latent_kl_log = np.zeros([n_batches, self.n_mdls])
            batch_lm_kl_log = np.zeros([n_batches, self.n_mdls])
            batch_mn_kl_log = np.zeros([n_batches, self.n_mdls])
            batch_psi_kl_log = np.zeros([n_batches, self.n_mdls])
            batch_s_kl_log = np.zeros([n_batches, self.n_mdls])

            for b_i in range(n_batches):

                obj = 0  # Keeps track of objective, summed across models
                optimizer.zero_grad()

                for m_i in range(self.n_mdls):
                    mdl_coll = self.vi_collections[m_i]
                    mdl_device = self.vi_collection_devices[m_i]
                    mdl_posteriors = mdl_coll.posteriors

                    # Move the posteriors to the appropriate device (this is neccesary if using shared posteriors)
                    mdl_posteriors.to(mdl_device)

                    batch_inds = batch_smp_inds[m_i][b_i]
                    n_batch_data_pts = len(batch_inds)

                    corr_f = float(model_n_data_pts[m_i])/n_batch_data_pts
                    elbo_vls_i = approximate_elbo(coll=mdl_coll, priors=self.priors, n_smps=1, inds=batch_inds,
                                                  corr_f=corr_f, skip_lm_kl=skip_lm_kl, skip_mn_kl=skip_mn_kl,
                                                  skip_psi_kl=skip_psi_kl, skip_s_kl=skip_s_kl)

                    nell = -1*elbo_vls_i['ell']
                    latent_kl = elbo_vls_i['latent_kl']
                    lm_kl = elbo_vls_i['lm_kl']
                    mn_kl = elbo_vls_i['mn_kl']
                    psi_kl = elbo_vls_i['psi_kl']
                    s_kl = elbo_vls_i['s_kl']

                    # Calculate gradients for this batch
                    mdl_obj = nell + latent_kl + lm_kl + mn_kl + psi_kl + s_kl
                    mdl_obj.backward()
                    obj += get_scalar_vl(mdl_obj)

                    # Log progress
                    batch_obj[b_i] = obj
                    batch_nell_log[b_i, m_i] = get_scalar_vl(nell)
                    batch_latent_kl_log[b_i, m_i] = get_scalar_vl(latent_kl)
                    batch_lm_kl_log[b_i, m_i] = get_scalar_vl(lm_kl)
                    batch_mn_kl_log[b_i, m_i] = get_scalar_vl(mn_kl)
                    batch_psi_kl_log[b_i, m_i] = get_scalar_vl(psi_kl)
                    batch_s_kl_log[b_i, m_i] = get_scalar_vl(s_kl)

                optimizer.step()

            # Enforce floor on private variances if estimating them with point estimates
            if psi_point_estimate:
                for coll in self.vi_collections:
                    enforce_floor(coll.mdl.psi, self.min_psi)

            # Update logs
            obj_log[e_i] = np.mean(batch_obj)
            nell_log[e_i, :] = np.mean(batch_nell_log, axis=0)
            latent_kl_log[e_i, :] = np.mean(batch_latent_kl_log, axis=0)
            lm_kl_log[e_i, :] = np.mean(batch_lm_kl_log, axis=0)
            mn_kl_log[e_i, :] = np.mean(batch_mn_kl_log, axis=0)
            psi_kl_log[e_i, :] = np.mean(batch_psi_kl_log, axis=0)
            s_kl_log[e_i, :] = np.mean(batch_s_kl_log, axis=0)

            if (update_int is not None) and (e_i % update_int == 0):
                t_now = time.time()
                self._print_status_update(epoch_i=e_i, obj_v=obj_log[e_i], nell_v=nell_log[e_i, :],
                                          latent_kl_v=latent_kl_log[e_i, :], lm_kl_v=lm_kl_log[e_i, :],
                                          mn_kl_v=mn_kl_log[e_i, :], psi_kl_v=psi_kl_log[e_i, :],
                                          s_kl_v=s_kl_log[e_i, :], lr=get_lr(optimizer), t=t_now - t_0)

            scheduler.step()

            # Create check point if needed
            if e_i in cp_epochs:
                self.save_checkpoint(epoch=e_i, save_folder=cp_save_folder, save_str=cp_save_str)

        # Generate final log structure
        log = {'obj': obj_log, 'nell': nell_log, 'latent_kl': latent_kl_log, 'lm_kl': lm_kl_log,
               'mn_kl': mn_kl_log, 'psi_kl': psi_kl_log, 's_kl': s_kl_log}

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
                subject_batch_smp_inds[b_i] = torch.tensor(perm_inds[start_smp_ind:end_smp_ind], dtype=torch.long,
                                                           device=self.vi_collection_devices[i])
                start_smp_ind = end_smp_ind
            batch_smp_inds[i] = subject_batch_smp_inds

        return batch_smp_inds

    def get_memory_usage(self) -> List[dict]:
        """ Returns amount of current memory used on each device used for fitting.

        Returns:
            memory_usage: memory_usage[i] contains the summary for device i as a dictionary with the following keys:

            type: The type of device - either 'cpu' or 'cuda'

            current: The current memory usage of the device in GB. If the device is cpu, this is the memory used by
            the process that called this function.

            max: The max memory used for the device.  If the device type is cpu, this will be nan.

            device: The device object the memory stats are for
        """

        # Make sure we ask for memory usaged on cpu if it is not listed in the devices
        cpu_listed = False
        for d in self.devices:
            if d.type == 'cpu':
                cpu_listed = True
                break

        if not cpu_listed:
            mem_devices = [torch.device('cpu')] + list(self.devices)
        else:
            mem_devices = self.devices

        # Get memory usage
        mem_summary = summarize_memory_stats(mem_devices)

        # Record the device with the memory stats
        for d_i, sum_d in enumerate(mem_summary):
            sum_d['device'] = mem_devices[d_i]

        return mem_summary

    @staticmethod
    def plot_log(log: dict):
        """ Generates a figure showing logged values. """

        POSSIBLE_FIELDS = ['obj', 'nell', 'latent_kl', 'lm_kl', 'mn_kl', 'psi_kl', 's_kl']
        FIELD_LABELS = ['Objective', 'NELL', 'Latent KL', 'LM KL', 'Mn KL', 'Psi KL', 'S KL']

        n_possible_fields = len(POSSIBLE_FIELDS)

        # See which fields are actually in the log
        present_fields = [True if k in list(log.keys()) else False for k in POSSIBLE_FIELDS]
        n_present_fields = np.sum(present_fields)
        n_rows = np.ceil(n_present_fields / 2.0).astype('int')

        plt.figure()

        cnt = 0
        for i in range(n_possible_fields):
            if present_fields[i] is True:
                cnt += 1
                ax = plt.subplot(n_rows, 2, cnt)
                ax.plot(log[POSSIBLE_FIELDS[i]])
                plt.xlabel('Epoch')
                plt.ylabel(FIELD_LABELS[i])

    def save_checkpoint(self, epoch: int, save_folder: StrOrPath, save_str: str = None):
        """ Saves a check point of models, posteriors and priors being fit.

        Everything will be saved after it has been moved to cpu.

        Args:

            epoch: The epoch that this checkpoint was created on.

            save_folder: The folder that checkpoints should be saved in.

            save_str: An optional string to add to the saved check point names.  Saved names will be
            of the format 'cp_<save_str>_<epoch>.pt'.
        """

        cp = {'vi_collections': [coll.generate_checkpoint() for coll in self.vi_collections],
              'priors': self.priors.generate_checkpoint(),
              'epoch': epoch}

        save_name = 'cp_' + save_str + str(epoch) + '.pt'
        save_path = pathlib.Path(save_folder) / save_name
        torch.save(cp, save_path)

        print('Saved check point for epoch ' + str(epoch) + '.')

    def _print_status_update(self, epoch_i, obj_v, nell_v, latent_kl_v, lm_kl_v, mn_kl_v, psi_kl_v, s_kl_v, lr, t):
        """ Prints a formatted status update to the screen. """

        print('')
        print('=========== EPOCH ' + str(epoch_i) + ' COMPLETE ===========')
        print('Obj: {:.2e}'.format(obj_v))
        print('----------------------------------------')
        if nell_v is not None:
            print('NELL: ' + list_to_str(nell_v))
        if latent_kl_v is not None:
            print('Latent KL: ' + list_to_str(latent_kl_v))
        print('LM KL: ' + list_to_str(lm_kl_v))
        print('Mn KL: ' + list_to_str(mn_kl_v))
        print('Psi KL: ' + list_to_str(psi_kl_v))
        print('S KL: ' + list_to_str(s_kl_v))
        print('----------------------------------------')
        print('LR: ' + str(lr))
        print('Elapsed time (secs): ' + str(t))

        # Print memory usage
        mem_stats = self.get_memory_usage()
        print('----------------------------------------')
        for stat_d in mem_stats:
            if stat_d['type'] == 'cpu':
                print('CPU cur memory used (GB): {:.2e}'.format(stat_d['current']))
            else:
                print('GPU_' + str(stat_d['device'].index) + ' cur memory used (GB): {:.2e}'.format(stat_d['current']) +
                      ', max memory used (GB): {:.2e}'.format(stat_d['max']))


class PosteriorCollection():
    """ Contains posteriors over the parameters of an GNLDR model.

    By holding all of the posteriors together, we gain convenience, when doing things such as moving the
    posteriors from one device to another or getting parameters.

    """

    def __init__(self, latent_post: SampleLatentsGaussianVariationalPosterior, lm_post: OptionalDistribution = None,
                 mn_post: OptionalDistribution = None, psi_post: OptionalDistribution = None,
                 s_post: OptionalDistribution = None):
        """ Creates a new PosteriorCollection.

        Args:

            latent_post: The posterior over latent variables for each sample.

            lm_post: The posterior over the coefficients of the loading matrix.

            mn_post: The posterior over the mean coefficients of the mean vector.

            psi_post: The posterior over private variances.

            s_post: The posterior over the scales parameters.

        """

        self.latent_post = latent_post
        self.lm_post = lm_post
        self.mn_post = mn_post
        self.psi_post = psi_post
        self.s_post = s_post

    @staticmethod
    def from_checkpont(cp: dict) -> 'PosteriorCollection':
        """ Generates a new PosteriorCollection from a checkpoint dictionary.

        Args:

            cp: The checkpoint dictionary

        Returns:

            collection: The new collection
        """

        return PosteriorCollection(latent_post=cp['latent_post'], lm_post=cp['lm_post'], mn_post=cp['mn_post'],
                                   psi_post=cp['psi_post'], s_post=cp['s_post'])

    def generate_checkpoint(self):
        """ Generates a check point of the collection.

        The posteriors will be returned on cpu.

        Returns:
            cp: A dictionary with the keys 'latent_post', 'lm_post', 'mn_post', 'psi_post' and 's_post' with the
            posteriors for the latents, loading matrices, mean and private variances, respectively.
        """

        cur_device = self.parameters()[0].device
        self.to('cpu')
        cp = {'latent_post': copy.deepcopy(self.latent_post),
              'lm_post': copy.deepcopy(self.lm_post),
              'mn_post': copy.deepcopy(self.mn_post),
              'psi_post': copy.deepcopy(self.psi_post),
              's_post': copy.deepcopy(self.s_post)}

        self.to(cur_device)

        return cp

    def to(self, device: Union[str, torch.device]):
        """ Moves the collection the the specified device."""

        self.latent_post.to(device)

        if self.lm_post is not None:
            self.lm_post.to(device)
        if self.mn_post is not None:
            self.mn_post.to(device)
        if self.psi_post is not None:
            self.psi_post.to(device)
        if self.s_post is not None:
            self.s_post.to(device)

    def parameters(self) -> List[torch.nn.Parameter]:
        """ Returns all parameters in all distributions in the collection. """

        latent_post_parameters = list(self.latent_post.parameters())

        if self.lm_post is not None:
            lm_post_parameters = list(self.lm_post.parameters())
        else:
            lm_post_parameters = []
        if self.mn_post is not None:
            mn_post_parameters = list(self.mn_post.parameters())
        else:
            mn_post_parameters = []

        if self.psi_post is not None:
            psi_post_parameters = list(self.psi_post.parameters())
        else:
            psi_post_parameters = []

        if self.s_post is not None:
            s_post_parameters = list(self.s_post.parameters())
        else:
            s_post_parameters = []

        return (latent_post_parameters + lm_post_parameters + mn_post_parameters + psi_post_parameters +
                s_post_parameters)


class PriorCollection():
    """ Contains conditional priors over the parameters of an FA model.

    By holding all of the priors together, we gain convenience, when doing things such as moving the
    priors from one device to another to getting the parameters of priors.

    """

    def __init__(self, lm_prior: OptionalDistribution = None, mn_prior: OptionalDistribution = None,
                 psi_prior: OptionalDistribution = None, s_prior: OptionalDistribution = None):
        """ Creates a new PriorCollection.

        Args:

            lm_prior: The conditional prior over the coefficients of the loading matrix.

            mn_prior: The conditional prior over the mean coefficients of the mean vector.

            psi_prior: The conditional prior over private variances.

            s_prior: The conditional prior over scale parameters
        """

        self.lm_prior = lm_prior
        self.mn_prior = mn_prior
        self.psi_prior = psi_prior
        self.s_prior = s_prior

    def device(self) -> torch.device:
        """ Returns the device the priors are on.

        This function assumes all priors are on the same device.

        Returns:

             device: The device the priors are on.  If there are no prior distributions in the collection, returns None.

        """
        params = self.parameters()
        if len(params) == 0:
            return None
        else:
            return params[0].device

    @staticmethod
    def from_checkpoint(cp: dict) -> 'PriorCollection':
        """ Generates a new PriorCollection from a checkpoint.

        Args:

            cp: The checkpoint dictionary

        Returns:

            coll: The new collection
        """

        return PriorCollection(lm_prior=cp['lm_prior'], mn_prior=cp['mn_prior'], psi_prior=cp['psi_prior'],
                               s_prior=cp['s_prior'])

    def generate_checkpoint(self):
        """ Generates a check point of the collection.

        The priors will be returned on cpu.

        Returns:
            cp: A dictionary with the keys 'lm_prior', 'mn_prior' and 'psi_prior' with the priors for the
            loading matrices, mean and private variances, respectively.
        """

        move = False
        if not ((self.lm_prior is None) and (self.mn_prior is None) and (self.psi_prior is None)
                and (self.s_prior is None)):
            move = True

        if move:
            cur_device = self.parameters()[0].device
            self.to('cpu')

        cp = {'lm_prior': copy.deepcopy(self.lm_prior),
              'mn_prior': copy.deepcopy(self.mn_prior),
              'psi_prior': copy.deepcopy(self.psi_prior),
              's_prior': copy.deepcopy(self.s_prior)}

        if move:
            self.to(cur_device)

        return cp

    def to(self, device: Union[str, torch.device]):
        """ Moves the collection the the specified device."""

        if self.lm_prior is not None:
            self.lm_prior.to(device)
        if self.mn_prior is not None:
            self.mn_prior.to(device)
        if self.psi_prior is not None:
            self.psi_prior.to(device)
        if self.s_prior is not None:
            self.s_prior.to(device)

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

        if self.s_prior is not None:
            s_parameters = list(self.s_prior.parameters())
        else:
            s_parameters = []

        return lm_parameters + mn_parameters + psi_parameters + s_parameters


class VICollection():
    """ A collection of objects necessary for fitting one GNLDR model with variational inference.

    This is a wrapper object that contains everything needed for fitting one GNLDR model: it contains the data
    the model is fit to, the properties associated with each of the observed random variables in the model,
    the posterior distribution over model parameters, and the GNLDR model object itself.

    By placing all of these entities into a single object, certain things, like moving data and models between
    devices can be made more convenient.

     """

    def __init__(self, data: torch.Tensor, props: torch.Tensor, mdl: GNLDRMdl, posteriors: PosteriorCollection):
        """ Creates a new VICollection object.

        Args:

            data: The data the model is fit to.  Of shape n_smps*n_obs_vars

            props: The properties associated with each of the observed random variables in the model.
            Of shape n_obs_vars*n_props

            mdl: The model object.  When creating this object, each parameter in the model for which posteriors
            will be fitted (instead of point estimates) should be set to None.

            posteriors: The collection of posteriors that will be fit over model parameters.

        """

        self.data = data
        self.props = props
        self.mdl = mdl
        self.posteriors = posteriors

    @staticmethod
    def from_checkpoint(cp: dict, data: torch.Tensor = None, props: torch.Tensor = None) -> 'VICollection':
        """ Generates a new VI Collection from a checkpoint dictionary.

        Args:

            cp: The checkpoint dictionary

            data: Optional data to add to the created VI Collection (as checkpoints don't contain data)

            props: Optional properties to add to the created VI Collection (as checkpoints don't contain properties)

        Return:

            coll: The new collection

        """
        return VICollection(data=data, props=props,
                            posteriors=PosteriorCollection.from_checkpont(cp['posteriors']),
                            mdl=cp['mdl'])

    def generate_checkpoint(self):
        """ Generates a check point of the collection.

        The check point will consist of the model and posteriors, but not data or properties.
        The model and posteriors will be returned on cpu.

        Returns:
            cp: A dictionary with the keys 'mdl' with the mdl and 'posteriors' with the posterior collection.
        """

        cur_device = self.mdl.log_2_pi.device

        self.mdl.to('cpu')
        cp = {'mdl': copy.deepcopy(self.mdl), 'posteriors': self.posteriors.generate_checkpoint()}
        self.mdl.to(cur_device)

        return cp

    def to(self, device: Union[str, torch.device], move_data: bool = True):
        """ Moves everything in the collection to a device.

        Args:
            device: The device to move everything to.

            move_data: True if data should be moved as well.

        """

        if move_data:
            self.data = self.data.to(device)

        self.props = self.props.to(device)
        self.mdl.to(device)
        self.posteriors.to(device)

    def parameters(self) -> List[torch.nn.Parameter]:
        """ Returns parameters in the FA model and all posteriors in the collection. """

        return list(self.mdl.parameters()) + self.posteriors.parameters()


def approximate_elbo(coll: VICollection, priors: PriorCollection, n_smps: int, min_psi: float = .0001,
                     inds: torch.Tensor = None, corr_f: float = 1.0, skip_lm_kl: bool = False,
                     skip_mn_kl: bool = False, skip_psi_kl: bool = False, skip_s_kl: bool = False):
    """ Approximates the ELBO for a single model via sampling.

    Calculations will be performed on whatever device the vi collection is on.

    Args:

        coll: The vi collection with the model, posteriors, properties and data for the subject.

        priors: The collection of priors over model parameters.

        n_smps: The number of samples to use when calculating the ELBO

        inds: Indices of data points in coll.data that we should compute the ELBO for.  If not provided, all
        data poitns will be used.

        corr_f: A correction factor to be applied if using a subset of data points for computing the ELBO.  This is
        useful when using mini-batches of data.  For example, if we call approximate_elbo for each minibatch, and in
        each minibatch we use only 25% of the data points, then we need to correct for this by increasing the weight of
        the expected log-liklihood and KL divergence for the latent values by a factor of 4.

        skip_lm_kl: True if KL divergence between the prior and posteriors over loading matrices should not be included
        in the ELBO

        skip_mn_kl: True if KL divergence between the prior and posteriors over means should not be included in the ELBO

        skip_psi_kl: True if KL divergence between the prior and posteriors over private noise variances should not be
        included in the ELBO

        skip_s_kl: True if KL divergence between the prior and posteriors over scales should not be included in the ELBO

    Returns:

        elbo_vls: A dictionary with the following keys:

            elbo: The value of the elbo

            ell: The expected log-likelihood of the data

            latent_kl: The kl divergence between the prior and posteriors over latent values

            lm_kl: The kl divergence between the prior and posterior over the loading matrix

            mn_kl: The kl divergence between the prior and posterior over the mean vector

            psi_kl: The kl divergence between the prior and posterior over the nosie variances

            s_kl: The kl divergence between the prior and posterior over the nosie variances

    """

    # Move the priors to the same device the VI collection is on
    compute_device = coll.posteriors.latent_post.mns.device
    orig_prior_device = priors.device()
    priors.to(compute_device)

    # Determine if there are any parameters for which point estimates are used in place of distributions
    lm_point_estimate = coll.mdl.lm is not None
    mn_point_estimate = coll.mdl.mn is not None
    psi_point_estimate = coll.mdl.psi is not None
    s_point_estimate = coll.mdl.s is not None

    data = coll.data
    posteriors = coll.posteriors
    props = coll.props
    mdl = coll.mdl

    if inds is None:
        inds = torch.arange(0, data.shape[0], dtype=torch.int64)

    # Approximate ELBO
    elbo = 0.0
    for s_i in range(n_smps):

        # Sample our posteriors
        latents_smp = posteriors.latent_post.sample(inds=inds)

        if not lm_point_estimate:
            # We produce samples in both compact and standard form.  Compact form is used for
            # computing KL divergences; standard form is used when computing likelihoods
            lm_compact_smp, lm_standard_smp = _sample_posterior(post=posteriors.lm_post, props=props)
        else:
            lm_standard_smp = None  # Passing in None to the appropriate functions will signal we use the
            # parameter stored in the model object

        if not mn_point_estimate:
            mn_compact_smp, mn_standard_smp = _sample_posterior(post=posteriors.mn_post, props=props)
            mn_standard_smp = mn_standard_smp.squeeze()
        else:
            mn_standard_smp = None

        if not psi_point_estimate:
            psi_compact_smp, psi_standard_smp = _sample_posterior(post=posteriors.psi_post, props=props)
            psi_standard_smp = psi_standard_smp.squeeze()

            # Enforce floor on sampled private variances
            enforce_floor(psi_compact_smp, min_psi)
            enforce_floor(psi_standard_smp, min_psi)
        else:
            psi_standard_smp = None

        if not s_point_estimate:
            s_compact_smp, s_standard_smp = _sample_posterior(post=posteriors.s_post, props=props)
            s_standard_smp = s_standard_smp.squeeze()
        else:
            s_standard_smp = None

        # Compute expected log-likelihood
        sel_data = data[inds, :].to(compute_device)
        ell = corr_f * torch.sum(mdl.cond_log_prob(z=latents_smp, x=sel_data, lm=lm_standard_smp,
                                                      mn=mn_standard_smp, psi=psi_standard_smp, s=s_standard_smp))

        # Compute KL divergences
        latent_kl = corr_f * posteriors.latent_post.kl_btw_standard_normal(inds=inds)

        if (not lm_point_estimate) and (not skip_lm_kl):
            lm_kl = torch.sum(posteriors.lm_post.kl(d_2=priors.lm_prior, x=props, smp=lm_compact_smp))
        else:
            lm_kl = 0
        if (not mn_point_estimate) and (not skip_mn_kl):
            mn_kl = torch.sum(posteriors.mn_post.kl(d_2=priors.mn_prior, x=props, smp=mn_compact_smp))
        else:
            mn_kl = 0
        if (not psi_point_estimate) and (not skip_psi_kl):
            psi_kl = torch.sum(posteriors.psi_post.kl(d_2=priors.psi_prior, x=props, smp=psi_compact_smp))
        else:
            psi_kl = 0
        if (not s_point_estimate) and (not skip_s_kl):
            s_kl = torch.sum(posteriors.s_post.kl(d_2=priors.s_prior, x=props, smp=s_compact_smp))
        else:
            s_kl = 0

        # Calculate elbo for this sample
        elbo += ell - latent_kl - lm_kl - mn_kl - psi_kl - s_kl

    elbo = elbo/n_smps

    # Move the priors back to whatever device they were on
    if orig_prior_device is not None:
        priors.to(orig_prior_device)

    return {'elbo': elbo, 'ell': ell, 'latent_kl': latent_kl, 'lm_kl': lm_kl, 'mn_kl': mn_kl, 'psi_kl': psi_kl,
            's_kl': s_kl}


def evaluate_check_points(cp_folder: StrOrPath, data: Sequence[torch.Tensor], props: Sequence[torch.Tensor],
                          n_smps: int, fit_opts: dict = None, elbo_opts: dict = None, device: torch.device = None):
    """ Evaluates checkpoints.

    Args:

        cp_folder: The folder with check points in it.

        data: The data to evaluate the checkpoints on. data[i] is the data for evaluating the i^th model.

    """

    cp_folder = pathlib.Path(cp_folder)

    if device is None:
        device = torch.device('cpu')

    if elbo_opts is None:
        elbo_opts = {}

    if fit_opts is None:
        fit_opts = {'n_epochs': 100, 'init_lr': .1, 'milestones': [50], 'update_int': None}

    # Find all check point files
    cp_files = glob.glob(str(cp_folder / 'cp*.pt'))
    n_cps = len(cp_files)

    # Approximates the optimal ELBO, holding priors and posteriors for model parameters fixed, for each check point
    n_mdls = len(data)
    cp_elbo = np.zeros([n_cps, n_mdls])
    cp_epochs = np.zeros(n_cps)
    for cp_i, cp_file in enumerate(cp_files):
        cp = torch.load(cp_file)
        cp_epochs[cp_i] = cp['epoch']

        priors = PriorCollection.from_checkpoint(cp['priors'])

        for m_i in range(n_mdls):

            # Set the data and properties of the VI collection
            coll_i = VICollection.from_checkpoint(cp=cp['vi_collections'][m_i], data=data[m_i], props=props[m_i])

            n_latent_vars = coll_i.posteriors.latent_post.mns.shape[1]

            # Infer latents for the data - here we learn new distributions only on the latent variables, leaving
            # distributions over model parameters untouched
            coll_i.posteriors.latent_post, _ = infer_latents(n_latent_vars=n_latent_vars, vi_collection=coll_i,
                                                             data=data[m_i], device=device, fit_opts=fit_opts)

            # Approximate value of the ELBO
            with torch.no_grad():
                coll_i.to(device)
                elbo_vls = approximate_elbo(coll=coll_i, priors=priors, n_smps=n_smps, **elbo_opts)
                cp_elbo[cp_i, m_i] = elbo_vls['elbo'].detach().cpu().numpy()

        print('Done with check point: ' + str(cp_i + 1) + ' of ' + str(n_cps) + '.')

    # Sort everything
    sort_order = np.argsort(cp_epochs)
    cp_epochs = cp_epochs[sort_order]
    cp_elbo = cp_elbo[sort_order, :]

    return cp_epochs, cp_elbo


def generate_simple_prior_collection(n_prop_vars: int, n_intermediate_latent_vars: int,
                                     min_gaussian_std: float = .01,
                                     min_gamma_conc_vl: float = 1.0, min_gamma_rate_vl: float = .01,
                                     lm_mn_w_init_std: float = .01, lm_std_w_init_std: float = .01,
                                     mn_mn_w_init_std: float = .01, mn_std_w_init_std: float = .01,
                                     psi_conc_f_w_init_std: float = 1, psi_rate_f_w_init_std: float = .01,
                                     psi_conc_bias_mn: float = 10.0, psi_rate_bias_mn: float = 10.0,
                                     s_mn: float = 1.0, s_std: float = .1) -> PriorCollection:
    """ Generates conditional priors where simple functions of properties generate distribution parameters.

    The conditional priors over the coefficients of the loading matrix and mean vectors will be Gaussian, with:

        1) Means which are a linear function of properties

        2) Standard deviations which are passed through linear function (of properties), the absolute value will
        be taken and then a fixed offset is added to prevent standard deviations from going below a certain bound.

    The conditional priors over the private variances will be Gamma distributions with:

        1) Concentration and rate parameters which are functions of the same form as (2) above.

    Args:
        n_prop_vars: The number of properties the distributions are conditioned on.

        n_intermediate_latent_vars: The number of latent variables in the intermediate space (after the latent
        variables have been transformed through m)

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
    lm_mn_f = torch.nn.Linear(in_features=n_prop_vars, out_features=n_intermediate_latent_vars, bias=True)
    lm_std_f = torch.nn.Sequential(torch.nn.Linear(in_features=n_prop_vars, out_features=n_intermediate_latent_vars,
                                                   bias=True),
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

    # Generate prior for scales
    s_mn_f = ConstantRealFcn(init_vl=torch.tensor([s_mn]), learnable_values=False)
    s_std_f = ConstantRealFcn(init_vl=torch.tensor([s_std]), learnable_values=False)
    s_prior = CondGaussianDistribution(mn_f=s_mn_f, std_f=s_std_f)

    return PriorCollection(lm_prior=lm_prior, mn_prior=mn_prior, psi_prior=psi_prior, s_prior=s_prior)


def generate_hypercube_prior_collection(n_intermediate_latent_vars: int, hc_params: dict, min_gaussian_std: float = .01,
                                        min_gamma_conc_vl: float = 1.0, min_gamma_rate_vl: float = .01,
                                        lm_mn_init: float = 0.0, lm_std_init: float = .1,
                                        mn_mn_init: float = 0.0, mn_std_init: float = .1,
                                        psi_conc_vl_init: float = 10.0,
                                        psi_rate_vl_init: float = 10.0,
                                        learnable_stds: bool = True,
                                        s_mn: float = 1.0, s_std: float = 1.0) -> PriorCollection:
    """ Generates conditional priors where sums of hypercube functions of properties generate distribution parameters.

    The conditional prior over the coefficients of the loading matrix will be Gaussian, with:

        1) Means which are a sum of hypercube functions

        2) Standard deviations which are passed through a sum of hypercube functions (of properties),
        and then passed through an exponential plus a fixed offset (to prevent standard deviations from
        going below a certain bound).

    The conditional prior over the coefficients of the mean vector will be Gaussian, with:

        1) Means which are a sum of hypercube functions

        2) Standard deviations which are passed through a sum of hypercube functions (of properties),
        and then passed through an absolute value plut a fixed offset (to prevent standard deviations from
        going below a certain bound).

    The conditional priors over the private variances will be Gamma distributions with:

        1) Concentration and rate parameters which are functions of the same form as the standard deviation
        functions for the mean.

    Args:
        n_latent_vars: The number of latent variables in the FA models.

        hc_params: The parameters to pass to SumOfTiledHyperCubeBasisFcns when creating the sum of tiled
        hyper cube basis functions objects.

        min_gaussian_std: The floor on the standard deviation for the distributions on the coefficients of the loading
        matrix and mean vector.

        min_gamma_conc_vl: The floor on values that the concentration parameter of the Gamma distributions can take
        on.

        min_gamma_rate_vl: The floor on values that rate parameter of the Gamma distributions can take on.

        lm_mn_init: The initial value for the mean of distributions over coefficients in the loading matrix. This
        will be a constant value for all conditioning input.

        lm_std_init: The initial value for the standard deviation of distributions over coefficients in the loading
        matrix. This will be a constant value for all conditioning input.

        mn_mn_init: The initial value for the mean of distributions over coefficients in the mean vector. This
        will be a constant value for all conditioning input.

        mn_std_init: The initial value for the standard deviation of distributions over coefficients in the mean
        vector. This will be a constant value for all conditioning input.

        psi_conc_vl_init: The initial value for the concentration parameter of the Gamma distributions over private
        variances.  This will be a constant value for all conditioning input.

        psi_rate_vl_init: The initial value for the rate parameter of the Gamma distributions over private
        variances.  This will be a constant value for all conditioning input.

        learnable_stds: True if the functions determining the standard deviations should have learnable parameters or
        not.  Setting this to false, results in conditional distributions with non-learnable standard deviations.

    Returns:

        priors: The generated collection of priors.
    """

    # Generate prior for loadings matrix
    lm_prior = CondMatrixHypercubePrior(n_cols=n_intermediate_latent_vars, mn_hc_params=hc_params,
                                        std_hc_params=hc_params, min_std=min_gaussian_std,
                                        mn_init=lm_mn_init, std_init=lm_std_init)

    if not learnable_stds:
        for dist in lm_prior.dists:
            dist.std_f[0].b_m.requires_grad = False

    # Generate prior for the mean
    mn_mn_f = SumOfTiledHyperCubeBasisFcns(**hc_params)
    mn_std_f = torch.nn.Sequential(SumOfTiledHyperCubeBasisFcns(**hc_params),
                                   FixedOffsetAbs(o=min_gaussian_std))

    n_active_bump_fcns = np.cumprod(hc_params['n_div_per_hc_side_per_dim'])[-1]
    mn_mn_f.b_m.data[:] = mn_mn_init/n_active_bump_fcns
    mn_std_f[0].b_m.data[:] = (mn_std_init - min_gaussian_std)/n_active_bump_fcns

    if not learnable_stds:
        mn_std_f[0].b_m.requires_grad = False

    mn_prior = CondGaussianDistribution(mn_f=mn_mn_f, std_f=mn_std_f)

    # Generate prior for private variances
    psi_conc_f = torch.nn.Sequential(SumOfTiledHyperCubeBasisFcns(**hc_params),
                                     FixedOffsetAbs(o=min_gamma_conc_vl))
    psi_rate_f = torch.nn.Sequential(SumOfTiledHyperCubeBasisFcns(**hc_params),
                                     FixedOffsetAbs(o=min_gamma_rate_vl))

    psi_conc_f[0].b_m.data[:] = (psi_conc_vl_init - min_gamma_conc_vl)/n_active_bump_fcns
    psi_rate_f[0].b_m.data[:] = (psi_rate_vl_init - min_gamma_rate_vl)/n_active_bump_fcns

    psi_prior = CondGammaDistribution(conc_f=psi_conc_f, rate_f=psi_rate_f)

    # Generate prior for scales
    s_mn_f = ConstantRealFcn(init_vl=torch.tensor([s_mn]), learnable_values=False)
    s_std_f = ConstantRealFcn(init_vl=torch.tensor([s_std]), learnable_values=False)
    s_prior = CondGaussianDistribution(mn_f=s_mn_f, std_f=s_std_f)

    return PriorCollection(lm_prior=lm_prior, mn_prior=mn_prior, psi_prior=psi_prior, s_prior=s_prior)


def generate_basic_posteriors(n_obs_vars: Sequence[int], n_smps: Sequence[int], n_latent_vars: int,
                              n_intermediate_latent_vars: int, lm_opts: OptionalDict = None,
                              mn_opts: OptionalDict = None, psi_opts: OptionalDict = None,
                              s_opts: OptionalDict = None) -> List[PosteriorCollection]:
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

        n_latent_vars: The number of latent variables in the models (the dimensionality of the low-d space)

        n_intermediate_latent_vars: The number of intermediate latent variables in the model (the dimensionality of
        the space after the latent variables have been transformed through m)

        lm_opts: Dictionary of options to provide to MatrixGaussianProductDistribution when creating the
        posteriors over loading matrices.  See that object for available options.

        mn_opts: Dictionary of options to provide to MatrixGaussianProductDistribution when creating the
        posteriors over mean vectors.  See that object for available options.

        psi_opts: Dictionary of options to provide to GammaProductDistribution when creating the
        posteriors over private variances.  See that object for available options.

        s_opts: Dictionary of options to provide to MatrixGaussianProductDistribution when creating the
        posteriors over mean vectors.  See that object for available options.

    Returns:

        post_collections: post_colletions[i] contains the posterior collections for subject i.

    """

    if lm_opts is None:
        lm_opts = dict()
    if mn_opts is None:
        mn_opts = dict()
    if psi_opts is None:
        psi_opts = dict()
    if s_opts is None:
        s_opts = dict()

    n_mdls = len(n_obs_vars)
    post_collections = [None]*n_mdls

    for i, n_i in enumerate(n_obs_vars):
        latent_post = SampleLatentsGaussianVariationalPosterior(n_latent_vars=n_latent_vars, n_smps=n_smps[i])
        lm_post = MatrixGaussianProductDistribution(shape=[n_i, n_intermediate_latent_vars], **lm_opts)
        mn_post = MatrixGaussianProductDistribution(shape=[n_i, 1], **mn_opts)
        psi_post = GammaProductDistribution(n_vars=n_i, **psi_opts)
        s_post = MatrixGaussianProductDistribution(shape=[n_i, 1], **s_opts)

        post_collections[i] = PosteriorCollection(latent_post=latent_post, lm_post=lm_post, mn_post=mn_post,
                                                  psi_post=psi_post, s_post=s_post)

    return post_collections


def infer_latents(n_latent_vars: int, vi_collection: VICollection, data: torch.Tensor, fit_opts: dict,
                  device: torch.device = None) -> Tuple[SampleLatentsGaussianVariationalPosterior, dict]:
    """ Infers latents, leaving posterior and prior distributions over model parameters unchanged.

    Args:

        n_latent_vars: The number of latent variables in the model

        vi_collection: A VI collection.  Nothing in this VI Colleciton will be changed by calling this function.  This
        is only provided so the function has access to properties as well as posteriors over model parameters.

        data: The data latents should be inferred for.

        fit_opts: Options to pass into the call to Fitter.fit().  See documentation of that function for more details.

        device: Device that will be used for inference.  If None, all fitting will be done on CPU.

    Returns:

        latent_post: The posterior over latents for the provided data.

        log: The log of the fitting

    """

    if device is None:
        device = [torch.device('cpu')]

    n_smps = data.shape[0]

    latent_post = SampleLatentsGaussianVariationalPosterior(n_latent_vars=n_latent_vars, n_smps=n_smps)

    # Generate posteriors and a vi collection for inference, copying by reference when things will not be changed to
    # save memory
    compute_posteriors = PosteriorCollection(latent_post=latent_post,
                                             lm_post=vi_collection.posteriors.lm_post,
                                             mn_post=vi_collection.posteriors.mn_post,
                                             psi_post=vi_collection.posteriors.psi_post,
                                             s_post=vi_collection.posteriors.s_post)

    compute_vi_collection = VICollection(data=data, props=vi_collection.props, mdl=vi_collection.mdl,
                                         posteriors=compute_posteriors)

    priors = PriorCollection(lm_prior=None, mn_prior=None, psi_prior=None)

    # Setup the fitter, setting min_psi to 0 so we make sure we don't unintentionally change any private variances
    fitter = Fitter(vi_collections=[compute_vi_collection], priors=priors, devices=[device], min_psi=0.0)
    fitter.distribute()
    log = fitter.fit(**fit_opts, optimize_only_latents=True)
    fitter.distribute(devices=[torch.device('cpu')])

    return latent_post, log


def _sample_posterior(post: CondVAEDistribution, props: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    compact_smp = post.sample(props)
    standard_smp = post.form_standard_sample(compact_smp)
    return compact_smp, standard_smp
