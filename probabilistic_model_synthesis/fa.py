""" Classes and functions for performing model synthesis over factor analysis models. """

import copy
import glob
import itertools
import math
import pathlib
import time
from typing import List, Tuple, Sequence, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sklearn.decomposition as decomposition
import torch
import torch.optim

from janelia_core.math.basic_functions import optimal_orthonormal_transform
from janelia_core.ml.extra_torch_modules import FixedOffsetAbs
from janelia_core.ml.extra_torch_modules import SumOfTiledHyperCubeBasisFcns
from janelia_core.visualization.matrix_visualization import cmp_n_mats
from janelia_core.ml.torch_distributions import CondGaussianDistribution
from janelia_core.ml.torch_distributions import CondMatrixHypercubePrior
from janelia_core.ml.torch_distributions import CondVAEDistribution
from janelia_core.ml.torch_distributions import MatrixGaussianProductDistribution
from janelia_core.ml.utils import summarize_memory_stats
from probabilistic_model_synthesis.distributions import CondGammaDistribution
from probabilistic_model_synthesis.distributions import GammaProductDistribution
from probabilistic_model_synthesis.utilities import enforce_floor
from probabilistic_model_synthesis.utilities import get_lr
from probabilistic_model_synthesis.utilities import get_scalar_vl
from probabilistic_model_synthesis.utilities import list_to_str


# Define type aliases
OptionalDevices = Union[Sequence[torch.device], None]
OptionalDict = Union[dict, None]
OptionalDistribution = Union[CondVAEDistribution, None]
OptionalTensor = Union[torch.Tensor, None]
StrOrPath = Union[pathlib.Path, str]


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

    def log_prob(self, x: torch.Tensor, lm: OptionalTensor = None, mn: OptionalTensor = None,
                      psi: OptionalTensor = None) -> torch.Tensor:
        """ Computes the log probability of observations.

        Args:
            x: Observed values of shape n_smps*n_obs_variables

            lm: If provided, uses this in place of the loading matrix parameter of the model.

            mn: If provided, uses this in place of the mean parameter of the model.

            psi: If provided, uses this in place of the psi parameter of the model.

        Returns:

            ll: The log-likelihood of each sample.  Of shape n_smps.
        """

        n_smps, n_obs_vars = x.shape

        if lm is None:
            lm = self.lm
        if mn is None:
            mn = self.mn
        if psi is None:
            psi = self.psi

        x_centered = x - mn
        cov_m = torch.matmul(lm, lm.t()) + torch.diag(psi)

        ll = (-.5*n_obs_vars*self.log_2_pi - .5*torch.logdet(cov_m)
              - .5*torch.sum((torch.matmul(x_centered, torch.inverse(cov_m))*x_centered), dim=1))

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

        return torch.randn(n_data_pts, self.m, device=self.c.device).mm(self.c.T) + self.mns[inds, :]

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

    def __init__(self, vi_collections: Sequence['VICollection'], priors: 'PriorCollection', devices: OptionalDevices = None,
                 min_psi: float = .0001):
        """ Creates a new Fitter object.

        Args:
            vi_collections: The VI collections to use for each model.

            priors: The collection of conditional priors to fit.

            min_psi: The minimum value that private variances can take on when sampling.  Sampled private variance
            values will be thresholded at this value.

        """
        self.vi_collections = vi_collections
        self.vi_collection_devices = None
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
            to these. If not provided, the devides already in the objects devices attribute will be used.
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
            skip_psi_kl: bool = False, update_int: int = 10, cp_epochs: Sequence[int] = None,
            cp_save_folder: StrOrPath = None, cp_save_str: str = '', optimize_only_latents: bool = False):
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

            cp_epochs: A sequence of epochs after which check point should be created.

            optimize_only_latents: If true, only the parameters of the distributions over latents are are optimized.
            This is useful when wanting to hold the distributions over model parameters fixed but infer latents for
            new data poitns.

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

        if cp_epochs is None:
            cp_epochs = []

        # Make sure we have distributed priors and vi collections across devices
        if self.vi_collection_devices is None:
            raise(RuntimeError('Distribute must be called before fit.'))

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

        if not optimize_only_latents:
            params = self.priors.parameters() + list(itertools.chain(*[coll.parameters()
                                                                       for coll in self.vi_collections]))
        else:
            params = list(itertools.chain(*[coll.posteriors.latent_post.parameters() for coll in self.vi_collections]))
            skip_mn_kl = True
            skip_lm_kl = True
            skip_psi_kl = True

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

            for b_i in range(n_batches):

                obj = 0 # Keeps track of objective, summed across models
                optimizer.zero_grad()

                for m_i in range(self.n_mdls):
                    mdl_device = self.vi_collection_devices[m_i]
                    mdl_posteriors = self.vi_collections[m_i].posteriors
                    mdl_props = self.vi_collections[m_i].props
                    fa_mdl = self.vi_collections[m_i].mdl

                    # Move the posteriors to the appropriate device (this is neccesary if using shared posteriors)
                    mdl_posteriors.to(mdl_device)

                    batch_inds = batch_smp_inds[m_i][b_i]
                    # Get the data for this minibatch and make sure it is on the correct device
                    data_b_i = self.vi_collections[m_i].data[batch_inds, :].to(mdl_device)
                    n_batch_data_pts = data_b_i.shape[0]

                    # Sample our posteriors
                    latents_smp = mdl_posteriors.latent_post.sample(inds=batch_inds)

                    if not lm_point_estimate:
                        # We produce samples in both compact and standard form.  Compact form is used for
                        # computing KL divergences; standard form is used when computing likelihoods
                        lm_compact_smp, lm_standard_smp = _sample_posterior(post=mdl_posteriors.lm_post, props=mdl_props)
                    else:
                        lm_standard_smp = None  # Passing in None to the appropriate functions will signal we use the
                                                # parameter stored in the FA model object

                    if not mn_point_estimate:
                        mn_compact_smp, mn_standard_smp = _sample_posterior(post=mdl_posteriors.mn_post, props=mdl_props)
                        mn_standard_smp = mn_standard_smp.squeeze()
                    else:
                        mn_standard_smp = None

                    if not psi_point_estimate:
                        psi_compact_smp, psi_standard_smp = _sample_posterior(post=mdl_posteriors.psi_post, props=mdl_props)
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
                    batch_obj[b_i] = obj
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
            obj_log[e_i] = np.mean(batch_obj)
            nell_log[e_i, :] = np.mean(batch_nell_log, axis=0)
            latent_kl_log[e_i, :] = np.mean(batch_latent_kl_log, axis=0)
            lm_kl_log[e_i, :] = np.mean(batch_lm_kl_log, axis=0)
            mn_kl_log[e_i, :] = np.mean(batch_mn_kl_log, axis=0)
            psi_kl_log[e_i, :] = np.mean(batch_psi_kl_log, axis=0)

            if e_i % update_int == 0:
                t_now = time.time()
                self._print_status_update(epoch_i=e_i, obj_v=obj_log[e_i], nell_v=nell_log[e_i, :],
                                          latent_kl_v=latent_kl_log[e_i, :], lm_kl_v=lm_kl_log[e_i, :],
                                          mn_kl_v=mn_kl_log[e_i, :], psi_kl_v=psi_kl_log[e_i, :],
                                          lr=get_lr(optimizer), t=t_now - t_0)

            scheduler.step()

            # Create check point if needed
            if e_i in cp_epochs:
                self.save_checkpoint(epoch=e_i, save_folder=cp_save_folder, save_str=cp_save_str)

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

    def optimize_priors(self, n_epochs: int, init_lr: float = .01, milestones: List[int] = None,
                        gamma: float = .1, update_int: int = 100):

        # Make sure we have distributed priors and vi collections across devices
        if self.vi_collection_devices is None:
            raise(RuntimeError('Distribute must be called before optimize priors.'))

        # If milestones are not provided, we set things up to use the initial learning rate the whole time
        if milestones is None:
            milestones = [n_epochs + 1]

        # Determine if there are any parameters we are estimating with point estimates
        lm_point_estimate = self.vi_collections[0].mdl.lm is not None
        mn_point_estimate = self.vi_collections[0].mdl.mn is not None
        psi_point_estimate = self.vi_collections[0].mdl.psi is not None

        # Gather all parameters we will be optimizing
        params = self.priors.parameters()

        # Make sure we have no duplicate parameters
        params = list(set(params))

        optimizer = torch.optim.Adam(params=params, lr=init_lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=gamma)

        # Set things up for logging
        obj_log = np.zeros(n_epochs)
        lm_kl_log = np.zeros([n_epochs, self.n_mdls])
        mn_kl_log = np.zeros([n_epochs, self.n_mdls])
        psi_kl_log = np.zeros([n_epochs, self.n_mdls])

        # Optimization loop
        t_0 = time.time()
        for e_i in range(n_epochs):

            optimizer.zero_grad()
            obj = 0  # Keeps track of objective, summed across models
            for m_i in range(self.n_mdls):
                mdl_device = self.vi_collection_devices[m_i]
                mdl_posteriors = self.vi_collections[m_i].posteriors
                mdl_props = self.vi_collections[m_i].props

                # Move the posteriors to the appropriate device (this is neccesary if using shared posteriors)
                mdl_posteriors.to(mdl_device)

                if not lm_point_estimate:
                    lm_compact_smp, _ = self._sample_posterior(post=mdl_posteriors.lm_post, props=mdl_props)
                    lm_kl = torch.sum(mdl_posteriors.lm_post.kl(d_2=self.priors.lm_prior, x=mdl_props,
                                                                smp=lm_compact_smp))
                else:
                    lm_kl = 0

                if not mn_point_estimate:
                    mn_compact_smp, _ = self._sample_posterior(post=mdl_posteriors.mn_post, props=mdl_props)
                    mn_kl = torch.sum(mdl_posteriors.mn_post.kl(d_2=self.priors.mn_prior, x=mdl_props,
                                                                smp=mn_compact_smp))
                else:
                    mn_kl = 0

                if not psi_point_estimate:
                    psi_compact_smp, _ = self._sample_posterior(post=mdl_posteriors.psi_post, props=mdl_props)
                    psi_kl = torch.sum(mdl_posteriors.psi_post.kl(d_2=self.priors.psi_prior, x=mdl_props,
                                                                  smp=psi_compact_smp))
                else:
                    psi_kl = 0

                # Calculate gradients for this batch
                mdl_obj = lm_kl + mn_kl + psi_kl
                mdl_obj.backward()
                obj += get_scalar_vl(mdl_obj)

                # Log progress
                obj_log[e_i] = obj
                lm_kl_log[e_i, m_i] = get_scalar_vl((lm_kl))
                mn_kl_log[e_i, m_i] = get_scalar_vl((mn_kl))
                psi_kl_log[e_i, m_i] = get_scalar_vl((psi_kl))

                optimizer.step()

            if e_i % update_int == 0:
                t_now = time.time()
                self._print_status_update(epoch_i=e_i, obj_v=obj_log[e_i], nell_v=None,
                                          latent_kl_v=None, lm_kl_v=lm_kl_log[e_i, :],
                                          mn_kl_v=mn_kl_log[e_i, :], psi_kl_v=psi_kl_log[e_i, :],
                                          lr=get_lr(optimizer), t=t_now - t_0)

            scheduler.step()

        # Generate final log structure
        log = {'obj': obj_log, 'lm_kl': lm_kl_log, 'mn_kl': mn_kl_log, 'psi_kl': psi_kl_log}

        return log

    @staticmethod
    def plot_log(log: dict):
        """ Generates a figure showing logged values. """

        POSSIBLE_FIELDS = ['obj', 'nell', 'latent_kl', 'lm_kl', 'mn_kl', 'psi_kl']
        FIELD_LABELS = ['Objective', 'NELL', 'Latent KL', 'LM KL', 'Mn KL', 'Psi KL']

        n_possible_fields = len(POSSIBLE_FIELDS)

        # See which fields are actually in the log
        present_fields = [True if k in list(log.keys()) else False for k in POSSIBLE_FIELDS]
        print(present_fields)
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

    def _print_status_update(self, epoch_i, obj_v, nell_v, latent_kl_v, lm_kl_v, mn_kl_v, psi_kl_v, lr, t):
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

    def generate_checkpoint(self):
        """ Generates a check point of the collection.

        The posteriors will be returned on cpu.

        Returns:
            cp: A dictionary with the keys 'latent_post', 'lm_post', 'mn_post' and 'psi_post' with the posteriors for
            the latents, loading matrices, mean and private variances, respectively.
        """

        cur_device = self.parameters()[0].device
        self.to('cpu')
        cp = {'latent_post': copy.deepcopy(self.latent_post),
              'lm_post': copy.deepcopy(self.lm_post),
              'mn_post': copy.deepcopy(self.mn_post),
              'psi_post': copy.deepcopy(self.psi_post)}

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

    def generate_checkpoint(self):
        """ Generates a check point of the collection.

        The priors will be returned on cpu.

        Returns:
            cp: A dictionary with the keys 'lm_prior', 'mn_prior' and 'psi_prior' with the priors for the
            loading matrices, mean and private variances, respectively.
        """

        move = False
        if not ((self.lm_prior is None) and (self.mn_prior is None) and (self.psi_prior is None)):
            move = True

        if move:
            cur_device = self.parameters()[0].device
            self.to('cpu')

        cp = {'lm_prior': copy.deepcopy(self.lm_prior),
              'mn_prior': copy.deepcopy(self.mn_prior),
              'psi_prior': copy.deepcopy(self.psi_prior)}

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
                     corr_f: float = 1.0, inds: torch.Tensor = None):
    """ Approximates the ELBO for a single model via sampling.

    Calculations will be performed on whatever device the vi collection is on

    Args:

        coll: The vi collection with the model, posteriors, properties and data for the subject.

        priors: The collection of priors over model parameters.

        n_smps: The number of samples to use when calculating the ELBO

    Returns:

        elbo: The approximated value of the ELBO.
    """

    # Move the priors to the same device the VI collection is on
    compute_device = coll.posteriors.latent_post.mns.device
    orig_prior_device = priors.device()
    priors.to(compute_device)

    # Determine if there are any parameters for which point estimates are used in place of distributions
    lm_point_estimate = coll.mdl.lm is not None
    mn_point_estimate = coll.mdl.mn is not None
    psi_point_estimate = coll.mdl.psi is not None

    data = coll.data
    posteriors = coll.posteriors
    props = coll.props
    fa_mdl = coll.mdl

    # Approximate ELBO
    if inds is None:
        inds = torch.range(0, data.shape[0])
    print('inds.dtype: ' + str(inds.dtype))

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
            # parameter stored in the FA model object

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

        # Compute expected log-likelihood
        sel_data = data[inds,:].to(compute_device)
        ell = corr_f * torch.sum(fa_mdl.cond_log_prob(z=latents_smp, x=sel_data, lm=lm_standard_smp,
                                                    mn=mn_standard_smp, psi=psi_standard_smp))

        # Compute KL divergences
        latent_kl = corr_f * posteriors.latent_post.kl_btw_standard_normal(inds=inds)

        if not lm_point_estimate:
            lm_kl = torch.sum(posteriors.lm_post.kl(d_2=priors.lm_prior, x=props, smp=lm_compact_smp))
        else:
            lm_kl = 0
        if not mn_point_estimate:
            mn_kl = torch.summ(posteriors.mn_post.kl(d_2=priors.mn_prior, x=props, smp=mn_compact_smp))
        else:
            mn_kl = 0
        if not psi_point_estimate:
            psi_kl = torch.sum(posteriors.psi_post.kl(d_2=priors.psi_prior, x=props, smp=psi_compact_smp))
        else:
            psi_kl = 0

        # Calculate elbo for this sample
        elbo += ell - latent_kl - lm_kl - mn_kl - psi_kl

    elbo = elbo/n_smps

    # Move the priors back to whatever device they were on
    if orig_prior_device is not None:
        priors.to(orig_prior_device)

    return {'elbo': elbo, 'latent_kl': latent_kl, 'lm_kl': lm_kl, 'mn_kl': mn_kl, 'psi_kl': psi_kl}


def evaluate_check_points(cp_folder: StrOrPath, data: Sequence[torch.Tensor], props: Sequence[torch.Tensor]):
    """ Evaluates checkpoints.

    Args:

        cp_folder: The folder with check points in it.

        data: The data to evaluate the checkpoints on. data[i] is the data for evaluating the i^th model.

    """

    cp_folder = pathlib.Path(cp_folder)

    # Find all check point files
    cp_files = glob.glob(str(cp_folder / 'cp*.pt'))
    n_cps = len(cp_files)

    # Evaluate the log-liklihood of data for each check point
    n_mdls = len(data)
    cp_ll = np.zeros([n_cps, n_mdls])
    for cp_i, cp_file in enumerate(cp_files):
        cp = torch.load(cp_file)
        for m_i in range(n_mdls):
            coll_i = cp['vi_collections'][m_i]
            with torch.no_grad():
                mdl_i = coll_i['mdl']

                if mdl_i.lm is None:
                    lm_i = coll_i['posteriors']['lm_post'](props[m_i])
                else:
                    lm_i = None

                if mdl_i.mn is None:
                    mn_i = coll_i['posteriors']['mn_post'](props[m_i]).squeeze()
                else:
                    mn_i = None

                if mdl_i.psi is None:
                    psi_i = coll_i['posteriors']['psi_post'].mode(props[m_i]).squeeze()
                else:
                    psi_i = None
            print('cp_' + str(cp_i) + ', mdl: ' + str(m_i))
            ll = mdl_i.log_prob(x=data[m_i], lm=lm_i, mn=mn_i, psi=psi_i, use_sklearn=True)
            cp_ll[cp_i, m_i] = np.sum(ll)/data[m_i].shape[0]

    return cp_ll


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


def generate_hypercube_prior_collection(n_latent_vars: int, hc_params: dict, min_gaussian_std: float = .01,
                                        min_gamma_conc_vl: float = 1.0, min_gamma_rate_vl: float = .01,
                                        lm_mn_init: float = 0.0, lm_std_init: float = .1,
                                        mn_mn_init: float = 0.0, mn_std_init: float = .1,
                                        psi_conc_vl_init: float = 10.0,
                                        psi_rate_vl_init: float = 10.0,
                                        learnable_stds: bool = True) -> PriorCollection:
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

        min_gamma_rate_vl: The floor on values that rate parameter of the folded Gamma distributions can take on.

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
    lm_prior = CondMatrixHypercubePrior(n_cols=n_latent_vars, mn_hc_params=hc_params,
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
        psi_post = GammaProductDistribution(n_vars=n_i, **psi_opts)

        post_collections[i] = PosteriorCollection(latent_post=latent_post, lm_post=lm_post, mn_post=mn_post,
                                                  psi_post=psi_post)

    return post_collections


def initialize_basic_posteriors(posteriors, data):
    """ Initializes posteriors created by the function generate_basic_posteriors.

    The purpose of this function is to just get the mean and standard deviations of the various
    distributions (those over means, private variance and the coefficients of the loading matrices) in
    the right ranges.  It will do this by fitting a standard FA model to the provided data for each model,
    taking the means and standard deviations of the various parameters (e.g, the mean and standard deviation
    of all the coefficients in the fit loading matrices) and then setting the mean and standard deviations
    of the posteriors for each coefficent of the parameter equal to these values.
    """

    std_gain = .01

    n_mdls = len(data)
    n_components = len(posteriors[0].lm_post.dists)

    for m_i, (posteriors_i, data_i) in enumerate(zip(posteriors, data)):

        # Fit a standard FA model to the data
        mdl = decomposition.FactorAnalysis(n_components=n_components)
        mdl.fit(data_i)

        # Set means and standard deviations of the posteriors
        n_vars = data_i.shape[1]

        # Initialize posteriors for means
        posteriors_i.mn_post.dists[0].mn_f.f.vl.data[:] = np.mean(mdl.mean_)
        posteriors_i.mn_post.dists[0].std_f.f.set_value(std_gain*np.std(mdl.mean_) * np.ones(n_vars))

        # Initialize posteriors for loading matrix
        for dist_i in posteriors_i.lm_post.dists:
            dist_i.mn_f.f.vl.data[:] = np.mean(mdl.components_)
            dist_i.std_f.f.set_value(std_gain*np.std(mdl.components_) * np.ones(n_vars))

            # Initialize posteriros for noise variances
        noise_stds = np.sqrt(mdl.noise_variance_)
        posteriors_i.psi_post.set_mean(torch.ones(n_vars) * np.mean(noise_stds))
        posteriors_i.psi_post.set_std(std_gain*torch.ones(n_vars) * np.std(noise_stds))

        # Give user some feedback
        print('Done initializing posteriors for model ' + str(m_i + 1) + ' of ' + str(n_mdls) + '.')


def orthonormalize(lm: np.ndarray, latents: np.ndarray = None, unit_len_columns: bool = True) -> Tuple[np.ndarray]:
    """ Applies orthonormalization to a loading matrix and latents.

    Args:

        lm: The loading matrix of the FA model, of shape n_obs_vars*n_latents

        latents: The loadings infered for the FA model, of shape n_smps*n_latents

        unit_len_columns: If true, the columns of the loading matrix will all have an l_2 norm of 1.  This corresponds
        to standard orthonormalization.  If false, the columns will be orthogonal but not of unit length.  This
        corresponds to a transformation that leaves the amount of variance in the latent space unchanged.

    Returns:

        lm_o: The orthonormalized loading matrix

        latents: The latents for the orthornormalized model
    """

    u, s, v = np.linalg.svd(lm, full_matrices=False)

    if unit_len_columns:
        lm_o = u
    else:
        lm_o = np.matmul(u, np.diag(s))

    if latents is not None:
        if unit_len_columns:
            latents_o = np.matmul(latents, np.matmul(np.diag(s), v).transpose())
        else:
            latents_o = np.matmul(latents, v.transpose())
    else:
        latents_o = None

    return lm_o, latents_o


def infer_latents(vi_collection: VICollection, data: torch.Tensor, fit_opts: dict,
                  devices: OptionalDevices = None) -> FAVariationalPosterior:
    """ Infers latents, leaving posterior and prior distributions over model parameters unchanged.

    Args:

        vi_collection: A VI collection.  Nothing in this VI Colleciton will be changed by calling this function.  This
        is only provided so the function has access to properties as well as posteriors over model parameters.

        data: The data latents should be inferred for.

        fit_opts: Options to pass into the call to Fitter.fit().  See documentation of that function for more details.

        devices: Devices that will be used for inference.  If None, all fitting will be done on CPU.

    Returns:

        latent_post: The posterior over latents for the provided data.

        log: The log of the fitting

    """

    n_smps = data.shape[0]
    n_latent_vars = vi_collection.posteriors.lm_post(vi_collection.props).shape[1]

    latent_post = FAVariationalPosterior(n_latent_vars=n_latent_vars, n_smps=n_smps)

    vi_collection = copy.deepcopy(vi_collection)
    vi_collection.data = data
    vi_collection.posteriors.latent_post = latent_post

    priors = PriorCollection(lm_prior=None, mn_prior=None, psi_prior=None)

    # Setup the fitter, setting min_psi to 0 so we make sure we don't unintentionally change any private variances
    fitter = Fitter(vi_collections=[vi_collection], priors=priors, devices=devices, min_psi=0.0)
    fitter.distribute()
    log = fitter.fit(**fit_opts, optimize_only_latents=True)
    fitter.distribute(devices=[torch.device('cpu')])

    return latent_post, log


def _sample_posterior(post: CondVAEDistribution, props: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    compact_smp = post.sample(props)
    standard_smp = post.form_standard_sample(compact_smp)
    return compact_smp, standard_smp
