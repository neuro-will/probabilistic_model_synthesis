""" Tools for performing model synthesis over non-linear regression models with Gaussian noise. """

import copy
import glob
import itertools
import math
import pathlib
import time
from typing import List, Optional, Sequence, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import numpy.linalg
import torch

from janelia_core.math.basic_functions import divide_into_nearly_equal_parts
from janelia_core.math.basic_functions import list_grid_pts
from janelia_core.ml.extra_torch_modules import DenseLNLNet
from janelia_core.ml.torch_distributions import CondVAEDistribution
from janelia_core.ml.torch_distributions import CondMatrixHypercubePrior
from janelia_core.ml.torch_distributions import MatrixGammaProductDistribution
from janelia_core.ml.torch_distributions import MatrixGaussianProductDistribution
from janelia_core.ml.utils import list_torch_devices
from janelia_core.ml.utils import summarize_memory_stats
from janelia_core.stats.regression import r_squared
from janelia_core.visualization.matrix_visualization import cmp_n_mats

from probabilistic_model_synthesis.gaussian_nonlinear_dim_reduction import _sample_posterior
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


def align_low_d_spaces(w_0: np.ndarray, s_in_0: np.ndarray, b_in_0: np.ndarray,
                              w_1: np.ndarray, s_in_1: np.ndarray, b_in_1: np.ndarray,
                              z_0: Optional[np.ndarray] = None, z_1: Optional[np.ndarray] = None) -> Tuple[np.ndarray]:
    """
    Aligns the low-d spaces of two models.

    Args:

        w_0: The weights from model 0

        w_1: The weights from model 1

        s_in_0: The s_in parameter from model 0

        s_in_1: The s_in parameter from model 1

        b_in_0: The b_in parameter from model 0

        b_in1: The b_in parameter from model 0

        z_0: Low-d projections of data from model 0.  This should be after input scales and biases are applied.

        z_1: Low-d projections of data from model 1.  This should be after input scales and biases are applied.

    """

    # Get aligned parameters for model 1
    b1_aligned = b_in_0
    s1_aligned = s_in_0
    t = numpy.linalg.lstsq(w_1, w_0, rcond=None)
    w1_aligned = np.matmul(w_0, t[0])

    # Get latents for aligned model 1
    if z_1 is not None:
        z1_rev = (z_1 - b_in_1)/s_in_1  # Remove biases and scales from model 1 projections
        z1_aligned = s1_aligned*np.matmul(z1_rev, t[0]) + b1_aligned
        return b1_aligned, s1_aligned, w1_aligned, t[0], z1_aligned
    else:
        return b1_aligned, s1_aligned, w1_aligned, t[0]


def approximate_elbo(coll: 'VICollection', priors: 'PriorCollection', n_smps: int, min_psi: float = .0001,
                     inds: torch.Tensor = None, corr_f: float = 1.0, skip_w_kl: bool = False,
                     skip_s_in_kl: bool = False, skip_b_in_kl: bool = False, skip_s_out_kl: bool = False,
                     skip_b_out_kl: bool = False, skip_psi_kl: bool = False) -> dict:
    """
    Approximates the elbo via sampling.

    Note: This function will move the priors to whatever device the VI Collection is on.

    Args:
        coll: The vi collection with the model, posteriors, properties and data for the subject.

        priors: The collection of priors over model parameters.

        n_smps: The number of samples to use when calculating the ELBO

        inds: Indices of data points in coll.data that we should compute the ELBO for.  If not provided, all
        data points will be used.

        corr_f: A correction factor to be applied if using a subset of data points for computing the ELBO.  This is
        useful when using mini-batches of data.  For example, if we call approximate_elbo for each minibatch, and in
        each minibatch we use only 25% of the data points, then we need to correct for this by increasing the weight of
        the expected log-liklihood and KL divergence for the latent values by a factor of 4.

        skip_w_kl: True if KL divergence between the prior and posteriors over weights should not be included in the
        ELBO

        skip_s_in_kl: True if KL divergence between the prior and posteriors over s_in parameters should not be included
        in the ELBO

        skip_b_in_kl: True if KL divergence between the prior and posteriors over b_in parameters should not be included
        in the ELBO

        skip_s_out_kl: True if KL divergence between the prior and posteriors over s_out parameters should not be included
        in the ELBO

        skip_b_out_kl: True if KL divergence between the prior and posteriors over b_out parameters should not be included
        in the ELBO

        skip_psi_kl: True if KL divergence between the prior and posteriors over psi parameters should not be included
        in the ELBO

    Returns:

        elbo_vls: A dictionary with the following keys:

            elbo: The value of the elbo

            ell: The expected log-likelihood of the data

            w_kl: The kl divergence between the prior and posteriors over weights

            s_in_kl: The kl divergence between the prior and posterior over the s_in parameters

            b_in_kl: The kl divergence between the prior and posterior over the b_in parameters

            s_out_kl: The kl divergence between the prior and posterior over the s_out parameters

            b_out_kl: The kl divergence between the prior and posterior over the b_out parameters

            psi_kl: The kl divergence between the prior and posterior over the noise variances

    """

    # Move the priors to the same device the VI collection is on
    compute_device = coll.parameters()[0].device
    priors.to(compute_device)

    # Determine if there are any parameters for which point estimates are used in place of distributions
    w_point_estimate = coll.mdl.w is not None
    s_in_point_estimate = coll.mdl.s_in is not None
    b_in_point_estimate = coll.mdl.b_in is not None
    s_out_point_estimate = coll.mdl.s_out is not None
    b_out_point_estimate = coll.mdl.b_out is not None
    psi_point_estimate = coll.mdl.psi is not None

    data = coll.data
    posteriors = coll.posteriors
    props = coll.props
    mdl = coll.mdl

    if inds is None:
        inds = torch.arange(0, data[0].shape[0], dtype=torch.int64)

    # Approximate elbo
    elbo = 0.0
    for s_i in range(n_smps):

        # Sample our posteriors
        if not w_point_estimate:
            # We produce samples in both compact and standard form.  Compact form is used for
            # computing KL divergences; standard form is used when computing likelihoods
            w_compact_smp, w_standard_smp = _sample_posterior(post=posteriors.w_post, props=props)
        else:
            w_standard_smp = None  # Passing in None to the appropriate functions will signal we use the
            # parameter stored in the model object

        if not s_in_point_estimate:
            s_in_compact_smp, s_in_standard_smp = _sample_posterior(post=posteriors.s_in_post, props=props)
            s_in_standard_smp = s_in_standard_smp.squeeze()
        else:
            s_in_standard_smp = None

        if not b_in_point_estimate:
            b_in_compact_smp, b_in_standard_smp = _sample_posterior(post=posteriors.b_in_post, props=props)
            b_in_standard_smp = b_in_standard_smp.squeeze()
        else:
            b_in_standard_smp = None

        if not s_out_point_estimate:
            s_out_compact_smp, s_out_standard_smp = _sample_posterior(post=posteriors.s_out_post, props=props)
            s_out_standard_smp = s_out_standard_smp.squeeze()
        else:
            s_out_standard_smp = None

        if not b_out_point_estimate:
            b_out_compact_smp, b_out_standard_smp = _sample_posterior(post=posteriors.b_out_post, props=props)
            b_out_standard_smp = b_out_standard_smp.squeeze()
        else:
            b_out_standard_smp = None

        if not psi_point_estimate:
            psi_compact_smp, psi_standard_smp = _sample_posterior(post=posteriors.psi_post, props=props)
            psi_standard_smp = psi_standard_smp.squeeze()

            # Enforce floor on sampled private variances
            enforce_floor(psi_compact_smp[0], min_psi)
            enforce_floor(psi_standard_smp, min_psi)
        else:
            psi_standard_smp = None

        # Compute expected log-likelihood
        sel_x_data = data[0][inds, :].to(compute_device)
        sel_y_data = data[1][inds, :].to(compute_device)
        ell = corr_f * torch.sum(mdl.cond_log_prob(x=sel_x_data, y=sel_y_data, w=w_standard_smp, s_in=s_in_standard_smp,
                                                   b_in=b_in_standard_smp, s_out=s_out_standard_smp,
                                                   b_out=b_out_standard_smp, psi=psi_standard_smp))

        # Compute KL divergences
        if (not w_point_estimate) and (not skip_w_kl):
            w_kl = torch.sum(posteriors.w_post.kl(d_2=priors.w_prior, x=props, smp=w_compact_smp))
        else:
            w_kl = 0

        if (not s_in_point_estimate) and (not skip_s_in_kl):
            s_in_kl = torch.sum(posteriors.s_in_post.kl(d_2=priors.s_in_prior, x=props, smp=s_in_compact_smp))
        else:
            s_in_kl = 0

        if (not b_in_point_estimate) and (not skip_b_in_kl):
            b_in_kl = torch.sum(posteriors.b_in_post.kl(d_2=priors.b_in_prior, x=props, smp=b_in_compact_smp))
        else:
            b_in_kl = 0

        if (not s_out_point_estimate) and (not skip_s_out_kl):
            s_out_kl = torch.sum(posteriors.s_out_post.kl(d_2=priors.s_out_prior, x=props, smp=s_out_compact_smp))
        else:
            s_out_kl = 0

        if (not b_out_point_estimate) and (not skip_b_out_kl):
            b_out_kl = torch.sum(posteriors.b_out_post.kl(d_2=priors.b_out_prior, x=props, smp=b_out_compact_smp))
        else:
            b_out_kl = 0

        if (not psi_point_estimate) and (not skip_psi_kl):
            psi_kl = torch.sum(posteriors.psi_post.kl(d_2=priors.psi_prior, x=props, smp=psi_compact_smp))
        else:
            psi_kl = 0

        # Calculate elbo for this sample
        elbo += ell - w_kl - s_in_kl - b_in_kl - s_out_kl - b_out_kl - psi_kl

    elbo = elbo/n_smps

    return {'elbo': elbo, 'ell': ell, 'w_kl': w_kl, 's_in_kl': s_in_kl, 'b_in_kl': b_in_kl, 's_out_kl': s_out_kl,
            'b_out_kl': b_out_kl, 'psi_kl': psi_kl}


def compare_weight_prior_dists(w0_prior: CondVAEDistribution, w1_prior: CondVAEDistribution,
                               dim_0_range: Sequence, dim_1_range: Sequence, n_pts_per_dim: Sequence):
    """ Visualizes two conditional distributions over weights, aligned to one another.

    The w1_prior will be aligned to the w0_prior.

    Note: The function assumes 2-d conditioning variables.

    Args:

        w0_prior: The prior we align to

        w1_prior: The prior we align

        dim_0_range: The range of values of the form [start, stop] for the first dimension of conditioning values we
        view means and standard deviations over.

        dim_1_range: The range of values of the form [start, stop] for the second dimension of conditioning values we
        view means and standard deviations over.

        n_pts_per_dim: The number of points per dimension of the form [n_dim_1_pts, n_dim_2_pts] we view conditioning
        values on for each dimension.
    """
    # Code for function starts here
    pts, dim_pts = list_grid_pts(grid_limits=np.asarray([dim_0_range, dim_1_range]), n_pts_per_dim=n_pts_per_dim)
    pts = torch.tensor(pts, dtype=torch.float)

    w0_mn = w0_prior(pts).detach().cpu().numpy()
    w1_mn = w1_prior(pts).detach().cpu().numpy()


    # Now get standard deviations
    if 'dists' in dir(w0_prior):
        w0_std = np.concatenate([d.std_f(pts).detach().cpu().numpy() for d in w0_prior.dists], axis=1)
    else:
        w0_std = w0_prior.std_f(pts).detach().cpu().numpy()

    if 'dists' in dir(w1_prior):
        w1_std = np.concatenate([d.std_f(pts).detach().cpu().numpy() for d in w1_prior.dists], axis=1)
    else:
        w1_std = w1_prior.std_f(pts).detach().cpu().numpy()

    # Align mean and standard deviations of distribution 1 to 2
    # t = numpy.linalg.lstsq(w1_mn, w0_mn, rcond=None)
    # print(t[0])
    #w1_mn_al = np.matmul(w1_mn, t[0])

    w1_mn_supp = np.concatenate([w1_mn, np.ones([w1_mn.shape[0], 1])], axis=1)
    t = numpy.linalg.lstsq(w1_mn_supp, w0_mn, rcond=None)
    t = t[0]
    w1_mn_al = np.matmul(w1_mn_supp, t)
    t_std = t[0:-1, :]

    w1_std_al = np.zeros(w1_std.shape)
    for i, std_i in enumerate(w1_std):
        covar_i = np.diag(std_i**2)
        covar_i_t = np.matmul(t_std.transpose(), np.matmul(covar_i, t_std))
        w1_std_al[i, :] = np.sqrt(np.diag(covar_i_t))

    # Generate the figure
    plt.figure()

    def _plot_image(n_rows, n_cols, im_i, im_pts, title_str):
        ax = plt.subplot(n_rows, n_cols, im_i)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        im = ax.imshow(im_pts.reshape(n_pts_per_dim), vmin=None, vmax=None, origin='lower')
        plt.colorbar(im, cax=cax, orientation='vertical')
        ax.set_title(title_str)

    # Now generate images for modes
    n_dims = w0_mn.shape[1]
    n_rows = n_dims
    cnt = 0
    for i in range(n_dims):
        cnt += 1
        _plot_image(n_rows, 4, cnt, w0_mn[:, i], 'W_0:' + str(i) + ' Mn')
        cnt += 1
        _plot_image(n_rows, 4, cnt, w1_mn_al[:, i], 'W_1:' + str(i) + ' Mn')
        cnt += 1
        _plot_image(n_rows, 4, cnt, w0_std[:, i], 'W_0:' + str(i) + ' Std')
        cnt += 1
        _plot_image(n_rows, 4, cnt, w1_std_al[:, i], 'W_1:' + str(i) + ' Std')


def eval_check_point_perf(cps: List[dict], eval_data: List[Tuple[torch.Tensor]],
                          subj_props: List[torch.Tensor], eval_device: torch.device = None) -> np.ndarray:
    """ Evaluates model performance across check points.

    Model performance is evaluated as the r-squared between true and predicted output. Predicted output is generated
    with posterior means for model parameters.

    Note: After evaluation, VI

    Args:
        cps: list of check points.  Each entry is a dictionary created by Fitter.save_checkpoint()

        eval_data: eval_data[i] is a tupe of (input, output) data to use for evaluating the model for subject i in
        the check points.

        subj_props: The properties for subject i in the check points.

        eval_device: The device evaluation should be performed on.  If none, evaluation will be performed on cpu.

    Returns:

        cp_perf: cp_perf[c_i, s_i] is the performance for check point c_i for subject s_i.
    """

    if eval_device is None:
        eval_device = torch.device('cpu')

    n_cp = len(cps)
    n_subjs = len(eval_data)

    cp_perf = np.ndarray([n_cp, n_subjs])

    for cp_i, cp in enumerate(cps):
        for s_i, (data_i, props_i) in enumerate(zip(eval_data, subj_props)):
            coll = VICollection.from_checkpoint(cp['vi_collections'][s_i], props=props_i.to(eval_device))
            coll.to(eval_device)
            with torch.no_grad():
                y_hat = predict(coll, data_i[0].to(eval_device)).detach().cpu().numpy()
            cp_perf[cp_i, s_i] = np.mean(r_squared(truth=data_i[1].numpy(), pred=y_hat))

    return cp_perf

class Fitter():
    """ Fits multiple GNLR models together, performing model synthesis.

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

            min_psi: The minimum value that noise variances can take on when sampling.  Sampled noise variances
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
            gamma: float = .1, optim_opts: dict = None, skip_w_kl: bool = False, skip_s_in_kl: bool = False,
            skip_b_in_kl: bool = False, skip_s_out_kl: bool = False, skip_b_out_kl: bool = False,
            skip_psi_kl: bool = False, update_int: int = 10, cp_epochs: Sequence[int] = None,
            cp_save_folder: StrOrPath = None, cp_save_str: str = '', prev_epochs: int = 0):
        """ Fits GNLDR models together.

        Args:
            n_epochs: The number of epochs to run fitting for.

            n_batches: The number of batches to break the data up into during each epoch.

            init_lr: The initial learning rate to start optimzation with

            milestones: A list of epochs at which to reduce the learning rate by a factor of gamma.  If not provided,
            the initial learning rate will be used the whole time.

            gamma: The factor to reduce the learning rate by at each milestone in milestones

            optim_opts: Extra options for creating the optimizer.  If None, no extra options will be provided.

            skip_w_kl: If true, kl divergences between posteriors and the prior for weights are not calculated.
            This can be safely set to true when all posteriors are the same shared conditional posterior (in which
            minimizing the KL divergence can be achieved trivially by setting the prior also equal to the shared
            conditional posterior).

            skip_s_in_kl: If true, kl divergences between posteriors and the prior for s_in parameters are not
            calculated.

            skip_b_in_kl: If true, kl divergences between posteriors and the prior for the b_in parameters are not
            calculated.

            skip_s_out_kl: If true, kl divergences between posteriors and the prior for the s_out parameters are not
            calculated.

            skip_b_out_kl: If true, kl divergences between posteriors and the prior for the b_out parameters are not
            calculated.

            skip_psi_kl: If true, kl divergences between posteriors and the prior for the psi parameters are not
            calculated.

            update_int: The number of epochs after which we provide the user with a status update.  If None,
            no updates will be printed.

            cp_epochs: A sequence of epochs after which check point should be created.

            cp_save_folder: A folder where check points should be saved

            cp_save_str: A string to add to the name of saved check point files (see save_checkpoint() for more details).

            prev_epochs: The previous number of epochs of fitting that have passed before calling fit for this
            round of fitting. This is only used to add metadata to saved checkpoints for keeping track of the total
            number of fitting epochs over (possibly) many rounds of fitting.

        Returns:

            log: A dictionary with the following keys:
                obj: A numpy array of shape n_epochs.  obj[e_i] is the objective value after epoch e_i

                nell: A numpy array of shape n_epochs*n_models.  obj[e_i,m_i] is the negative expected log likelihood,
                averaged across batches after epoch e_i for model m_i

                w_kl: A numpy array of shape n_epochs*n_models.  w_kl[e_i,m_i] is the kl divergence
                between the prior distribution over weights and the posterior averaged across batches after epoch e_i
                for model m_i

                s_in_kl: A numpy array of shape n_epochs*n_models.  s_in_kl[e_i,m_i] is the kl divergence
                between the prior distribution over s_in parameters and the posterior averaged across batches after
                epoch e_i for model m_i

                b_in_kl: A numpy array of shape n_epochs*n_models.  b_in_kl[e_i,m_i] is the kl divergence
                between the prior distribution over b_in parameters and the posterior averaged across batches after
                epoch e_i for model m_i

                s_out_kl: A numpy array of shape n_epochs*n_models.  s_out_kl[e_i,m_i] is the kl divergence
                between the prior distribution over s_out parameters and the posterior averaged across batches after
                epoch e_i for model m_i

                b_out_kl: A numpy array of shape n_epochs*n_models.  b_out_kl[e_i,m_i] is the kl divergence
                between the prior distribution over b_out parameters and the posterior averaged across batches after
                epoch e_i for model m_i

                psi_kl: A numpy array of shape n_epochs*n_models.  psi_kl[e_i,m_i] is the kl divergence
                between the prior distribution over private variances and the posterior averaged across batches after
                epoch e_i for model m_i
        """

        if cp_epochs is None:
            cp_epochs = []

        if optim_opts is None:
            optim_opts = dict()

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

        params = self.priors.parameters() + list(itertools.chain(*[coll.parameters() for coll in self.vi_collections]))

        # Make sure we have no duplicate parameters
        params = list(set(params))

        optimizer = torch.optim.Adam(params=params, lr=init_lr, **optim_opts)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=gamma)

        # Determine the total number of data points we have for fitting each model
        model_n_data_pts = [coll.data[0].shape[0] for coll in self.vi_collections]

        # Set things up for logging
        obj_log = np.zeros(n_epochs)
        nell_log = np.zeros([n_epochs, self.n_mdls])
        w_kl_log = np.zeros([n_epochs, self.n_mdls])
        s_in_kl_log = np.zeros([n_epochs, self.n_mdls])
        b_in_kl_log = np.zeros([n_epochs, self.n_mdls])
        s_out_kl_log = np.zeros([n_epochs, self.n_mdls])
        b_out_kl_log = np.zeros([n_epochs, self.n_mdls])
        psi_kl_log = np.zeros([n_epochs, self.n_mdls])

        # Optimization loop
        t_0 = time.time()
        for e_i in range(n_epochs):

            # Determine which samples we use for each batch for each subject
            batch_smp_inds = self.generate_batch_smp_inds(n_batches)

            # Create variables for logging results across batches
            batch_obj = np.zeros(n_batches)
            batch_nell_log = np.zeros([n_batches, self.n_mdls])
            batch_w_kl_log = np.zeros([n_batches, self.n_mdls])
            batch_s_in_kl_log = np.zeros([n_batches, self.n_mdls])
            batch_b_in_kl_log = np.zeros([n_batches, self.n_mdls])
            batch_s_out_kl_log = np.zeros([n_batches, self.n_mdls])
            batch_b_out_kl_log = np.zeros([n_batches, self.n_mdls])
            batch_psi_kl_log = np.zeros([n_batches, self.n_mdls])

            for b_i in range(n_batches):

                obj = 0  # Keeps track of objective, summed across models
                optimizer.zero_grad()

                for m_i in range(self.n_mdls):
                    mdl_coll = self.vi_collections[m_i]
                    mdl_device = self.vi_collection_devices[m_i]
                    mdl_posteriors = mdl_coll.posteriors

                    # Move the collection to the appropriate device (this is necessary if using shared posteriors or
                    # if the models themselves share any components (such as a shared m module)
                    mdl_coll.to(mdl_device)
                    #mdl_posteriors.to(mdl_device)

                    batch_inds = batch_smp_inds[m_i][b_i]
                    n_batch_data_pts = len(batch_inds)

                    corr_f = float(model_n_data_pts[m_i])/n_batch_data_pts
                    elbo_vls_i = approximate_elbo(coll=mdl_coll, priors=self.priors, n_smps=1, inds=batch_inds,
                                                  corr_f=corr_f, skip_w_kl=skip_w_kl, skip_s_in_kl=skip_s_in_kl,
                                                  skip_b_in_kl=skip_b_in_kl, skip_s_out_kl=skip_s_out_kl,
                                                  skip_b_out_kl=skip_b_out_kl, skip_psi_kl=skip_psi_kl)

                    nell = -1*elbo_vls_i['ell']
                    w_kl = elbo_vls_i['w_kl']
                    s_in_kl = elbo_vls_i['s_in_kl']
                    b_in_kl = elbo_vls_i['b_in_kl']
                    s_out_kl = elbo_vls_i['s_out_kl']
                    b_out_kl = elbo_vls_i['b_out_kl']
                    psi_kl = elbo_vls_i['psi_kl']

                    # Calculate gradients for this batch
                    mdl_obj = nell + w_kl + s_in_kl + b_in_kl + s_out_kl + b_out_kl + psi_kl
                    mdl_obj.backward()
                    obj += get_scalar_vl(mdl_obj)

                    # Log progress
                    batch_obj[b_i] = obj
                    batch_nell_log[b_i, m_i] = get_scalar_vl(nell)
                    batch_w_kl_log[b_i, m_i] = get_scalar_vl(w_kl)
                    batch_s_in_kl_log[b_i, m_i] = get_scalar_vl(s_in_kl)
                    batch_b_in_kl_log[b_i, m_i] = get_scalar_vl(b_in_kl)
                    batch_s_out_kl_log[b_i, m_i] = get_scalar_vl(s_out_kl)
                    batch_b_out_kl_log[b_i, m_i] = get_scalar_vl(b_out_kl)
                    batch_psi_kl_log[b_i, m_i] = get_scalar_vl(psi_kl)

                optimizer.step()

            # Enforce floor on private variances if estimating them with point estimates
            if psi_point_estimate:
                for coll in self.vi_collections:
                    enforce_floor(coll.mdl.psi, self.min_psi)

            # Update logs
            obj_log[e_i] = np.mean(batch_obj)
            nell_log[e_i, :] = np.mean(batch_nell_log, axis=0)
            w_kl_log[e_i, :] = np.mean(batch_w_kl_log, axis=0)
            s_in_kl_log[e_i, :] = np.mean(batch_s_in_kl_log, axis=0)
            b_in_kl_log[e_i, :] = np.mean(batch_b_in_kl_log, axis=0)
            s_out_kl_log[e_i, :] = np.mean(batch_s_out_kl_log, axis=0)
            b_out_kl_log[e_i, :] = np.mean(batch_b_out_kl_log, axis=0)
            psi_kl_log[e_i, :] = np.mean(batch_psi_kl_log, axis=0)

            if (update_int is not None) and (e_i % update_int == 0):
                t_now = time.time()
                self._print_status_update(epoch_i=e_i, obj_v=obj_log[e_i], nell_v=nell_log[e_i, :],
                                          w_kl_v=w_kl_log[e_i, :], s_in_kl_v=s_in_kl_log[e_i, :],
                                          b_in_kl_v=b_in_kl_log[e_i, :], s_out_kl_v=s_out_kl_log[e_i, :],
                                          b_out_kl_v=b_out_kl_log[e_i, :], psi_kl_v=psi_kl_log[e_i, :],
                                          lr=get_lr(optimizer), t=t_now - t_0)

            scheduler.step()

            # Create check point if needed
            if e_i in cp_epochs:
                self.save_checkpoint(epoch=e_i, save_folder=cp_save_folder, save_str=cp_save_str,
                                     prev_epochs=prev_epochs)

        # Generate final log structure
        log = {'obj': obj_log, 'nell': nell_log, 'w_kl': w_kl_log, 's_in_kl': s_in_kl_log, 'b_in_kl': b_in_kl_log,
               's_out_kl': s_out_kl_log, 'b_out_kl': b_out_kl_log, 'psi_kl': psi_kl_log}

        return log

    def generate_batch_smp_inds(self, n_batches: int) -> List[List[np.ndarray]]:
        """ Generates indices of random mini-batches of samples for each model.

        Args:
            n_batches: The number of batches to break the data up for each subject into.

        Returns:
            batch_smp_inds: batch_smp_inds[i][j] is the sample indices for the j^th batch for the i^th model

        """

        n_smps = [self.vi_collections[i].data[0].shape[0] for i in range(self.n_mdls)]

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

        # Make sure we ask for memory usage on cpu if it is not listed in the devices
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

        POSSIBLE_FIELDS = ['obj', 'nell', 'w_kl', 's_in_kl', 'b_in_kl', 's_out_kl', 'b_out_kl', 'psi_kl']
        FIELD_LABELS = ['Objective', 'NELL', 'W KL', 's_in KL', 'b_in KL', 's_out KL', 'b_out_kl', 'Psi KL']

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

    def save_checkpoint(self, epoch: int, save_folder: StrOrPath, save_str: str = None, prev_epochs: int = 0):
        """ Saves a check point of models, posteriors and priors being fit.

        Everything will be saved after it has been moved to cpu.

        Args:

            epoch: The epoch that this checkpoint was created on.

            save_folder: The folder that checkpoints should be saved in.

            save_str: An optional string to add to the saved check point names.  Saved names will be
            of the format 'cp_<save_str><total_epoch>.pt', where total_epoch = prev_epochs + epoch

            prev_epochs: An offset to add to epoch numbers to create a "total epoch" number. This is useful if
            performing multiple rounds of fitting and wanting to keep track of the total number of epochs that
            have passed over all fitting rounds.

        """

        cp = {'vi_collections': [coll.generate_checkpoint() for coll in self.vi_collections],
              'priors': self.priors.generate_checkpoint(),
              'epoch': epoch,
              'total_epoch': prev_epochs + epoch}

        save_name = 'cp_' + save_str + str(prev_epochs + epoch) + '.pt'
        save_path = pathlib.Path(save_folder) / save_name
        torch.save(cp, save_path)

        print('Saved check point for epoch ' + str(epoch) + '.')

    def _print_status_update(self, epoch_i, obj_v, nell_v, w_kl_v, s_in_kl_v, b_in_kl_v, s_out_kl_v, b_out_kl_v,
                             psi_kl_v, lr, t):
        """ Prints a formatted status update to the screen. """

        print('')
        print('=========== EPOCH ' + str(epoch_i) + ' COMPLETE ===========')
        print('Obj: {:.2e}'.format(obj_v))
        print('----------------------------------------')
        if nell_v is not None:
            print('NELL: ' + list_to_str(nell_v))
        print('W KL: ' + list_to_str(w_kl_v))
        print('S_in KL: ' + list_to_str(s_in_kl_v))
        print('B_in KL: ' + list_to_str(b_in_kl_v))
        print('S_out KL: ' + list_to_str(s_out_kl_v))
        print('B_out KL: ' + list_to_str(b_out_kl_v))
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


class GNLRMdl(torch.nn.Module):
    """ A nonlinear regression model with Gaussian noise on predicted variables.

    A Gaussian nonlinear regression (GNLR) model can be described by the following generative process, conditioned
    on input x_t \in R^n:

    First, mean values of predicted variables are generated according to:

        z_t = s_in \had_prod w*x_t + b_in, for w \in R^{p \times n}, b_in \in R^p, s_in \in R^p where * denotes
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

    def __init__(self, m: torch.nn.Module, w: OptionalTensor = None, s_in: OptionalTensor = None,
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

            w: The weights projecting into the low-d space.  Of shape n_input_vars*p

            s_in: The scales for projecting into the low-d space. Of shape p

            b_in: The biases for projecting into the low-d space. Of shape p

            s_out: The scales for mapping to the the means of predicted variables. Of shape n_predicted_variables

            b_out: The biases for projecting into the means of predicted variables. Of shape n_predicted variables

            psi: A vector of noise variances of for the predicted variables

        """
        super().__init__()

        self.m = m

        if w is not None:
            self.w = torch.nn.Parameter(w)
        else:
            self.w = None

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
            self.s_out = None

        if b_out is not None:
            self.b_out = torch.nn.Parameter(b_out)
        else:
            self.b_out = None

        if psi is not None:
            self.psi = torch.nn.Parameter(psi)
        else:
            self.psi = None

        self.register_buffer('log_2_pi', torch.log(torch.tensor(2 * math.pi)))

    @staticmethod
    def compare_mdls(m0: 'GNLRMdl', m1: 'GNLRMdl'):
        """ Visually compares two models.

        Args:

            m1: The first model

            m2: The second model
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

        w_0 = m0.w.detach().cpu().numpy()
        w_1 = m1.w.detach().cpu().numpy()
        s_in_0 = m0.s_in.detach().cpu().numpy()
        s_in_1 = m1.s_in.detach().cpu().numpy()
        b_in_0 = m0.b_in.detach().cpu().numpy()
        b_in_1 = m1.b_in.detach().cpu().numpy()
        s_out_0 = m0.s_out.detach().cpu().numpy()
        s_out_1 = m1.s_out.detach().cpu().numpy()
        b_out_0 = m0.b_out.detach().cpu().numpy()
        b_out_1 = m1.b_out.detach().cpu().numpy()
        psi_0 = m0.psi.detach().cpu().numpy()
        psi_1 = m1.psi.detach().cpu().numpy()

        # Align w_1 to w_0
        t = numpy.linalg.lstsq(w_1, w_0, rcond=None)
        w_1_aligned = np.matmul(w_1, t[0])
        print(t[0])

        # Make plots of scalar variables
        _make_subplot([0, 0], 3, 3, s_in_0, s_in_1, 's_in')
        _make_subplot([0, 6], 3, 3, b_in_0, b_in_1, 'b_in')
        _make_subplot([8, 0], 3, 3, s_out_0, s_out_1, 's_out')
        _make_subplot([8, 6], 3, 3, b_out_0, b_out_1, 'b_out')
        _make_subplot([16, 0], 3, 3, psi_0, psi_1, 'Psi')

        # Make plots of weights
        w_diff = w_0 - w_1_aligned

        w1_grid_info = {'grid_spec': grid_spec}
        w1_cell_info = list()
        w1_cell_info.append({'loc': [0, 10], 'rowspan': 24, 'colspan': 3})
        w1_cell_info.append({'loc': [0, 15], 'rowspan': 24, 'colspan': 3})
        w1_cell_info.append({'loc': [0, 20], 'rowspan': 24, 'colspan': 3})
        w1_grid_info['cell_info'] = w1_cell_info
        cmp_n_mats([w_0, w_1_aligned, w_diff], show_colorbars=True, titles=['W_0', 'W_1', 'W_0 -  W_1'],
                   grid_info=w1_grid_info)

    def cond_log_prob(self, x: torch.Tensor, y: torch.Tensor, w: OptionalTensor = None, s_in: OptionalTensor = None,
                      b_in: OptionalTensor = None, s_out: OptionalTensor = None, b_out: OptionalTensor = None,
                      psi: OptionalTensor = None) -> torch.Tensor:
        """ Computes the log probability of predicted variables given input.

        Args:
            x: Input into the regression model of shape n_smps*n_input_variables

            y: Predicted values of shape n_smps*n_predicted_variables

            w: If provided, this is used in place of the w_in parameter of the model

            s_in: If provided, this is used in place of the s_in parameter of the model

            b_in: If provided, this is used in place of the b_in parameter of the model

            s_out: If provided, this is used in place of the s_out parameter of the model

            b_out: If provided, this is used in place of the b_out parameter of the model

            psi: If provided, this is used in place of the psi parameter of the model

        Returns:

            ll: The log-likelihood for each sample.  Of shape n_smps.

        """

        n_smps, n_pred_vars = y.shape

        if w is None:
            w = self.w
        if b_in is None:
            b_in = self.b_in
        if s_in is None:
            s_in = self.s_in
        if s_out is None:
            s_out = self.s_out
        if b_out is None:
            b_out = self.b_out
        if psi is None:
            psi = self.psi

        mns = self.cond_mean(x=x, w=w, s_in=s_in, b_in=b_in, s_out=s_out, b_out=b_out)

        ll = -.5*torch.sum(((y - mns)**2)/psi, dim=1)
        ll -= .5*torch.sum(torch.log(psi))
        ll -= .5*n_pred_vars*self.log_2_pi

        return ll

    def cond_mean(self, x: torch.Tensor, w: OptionalTensor = None, s_in: OptionalTensor = None,
                  b_in: OptionalTensor = None, s_out: OptionalTensor = None,
                  b_out: OptionalTensor = None) -> torch.Tensor:
        """ Computes mean of predicted variables given input.

        Args:
            x: Input into the regression model of shape n_smps*n_input_variables

            w: If provided, this is used in place of the w_in parameter of the model

            s_in: If provided, this is used in place of the s_in parameter of the model

            b_in: If provided, this is used in place of the b_in parameter of the model

            s_out: If provided, this is used in place of the s_out parameter of the model

            b_out: If provided, this is used in place of the b_out parameter of the model

        Returns:

            mn: Means of predicted variables conditioned on input. Of shape n_smps*n_predicted variables

        """
        if w is None:
            w = self.w
        if b_in is None:
            b_in = self.b_in
        if s_in is None:
            s_in = self.s_in
        if s_out is None:
            s_out = self.s_out
        if b_out is None:
            b_out = self.b_out

        z = self.project(x=x, w=w, s_in=s_in, b_in=b_in, apply_scales_and_biases=True)
        l = self.m(z)
        return s_out*l + b_out

    def fit(self, x: torch.Tensor, y: torch.Tensor, n_epochs: int, n_mini_batches: int, fit_opts: dict = None,
            update_int: int = 100) -> np.ndarray:
        """ Fits a model to data.

        Note: This function assumes that all parameter attributes of the object have been assigned.

        Args:

            x: Training input data, of shape n_smps*n_input_variables

            y: Training output data, of shape n_smps*n_output_variables

            n_epochs: The number of epochs to run training for

            n_mini_batches: The number of mini-batches to use per epoch

            fit_opts: Dictionary of options for creating the Adam optimizer.

            update_int: The frequency (in number of epochs) that updates should be printed to screen

        Returns:

            nll_log: The negative log-likelihood evaluated after each epoch

        """

        if ((self.w is None) or (self.b_in is None) or (self.s_in is None) or (self.s_out is None)
                or (self.b_out is None) or (self.psi is None)):
            raise(ValueError('All parameter attributes of this object must be assigned before calling fit.'))

        n_s = x.shape[0]
        if n_s < n_mini_batches:
            raise(ValueError('Only ' + str(n_s) + ' samples in trianing data, but ' + str(n_mini_batches) + ' requested.'))

        optimizer = torch.optim.Adam(params=self.parameters(), **fit_opts)

        nll_log = np.zeros(n_epochs)
        for e_i in range(n_epochs):

            # Assign data to batches for this epoch
            mini_batch_sizes = divide_into_nearly_equal_parts(n_s, n_mini_batches)
            perm_inds = np.random.permutation(n_s)
            start_mini_batch_ind = 0
            mini_batch_inds = [None]*n_mini_batches
            for b_i in range(n_mini_batches):
                end_mini_batch_ind = start_mini_batch_ind + mini_batch_sizes[b_i]
                mini_batch_inds[b_i] = torch.tensor(perm_inds[start_mini_batch_ind:end_mini_batch_ind],
                                                    dtype=torch.long, device=x.device)
                start_mini_batch_ind = end_mini_batch_ind

            # Run this batch
            for b_i in range(n_mini_batches):

                mb_x = x[mini_batch_inds[b_i], :]
                mb_y = y[mini_batch_inds[b_i], :]

                optimizer.zero_grad()

                # Calculate objective - making sure to account for batch size
                nll = -1*(n_s/mini_batch_sizes[b_i])*torch.sum(self.cond_log_prob(x=mb_x, y=mb_y))
                nll.backward()

                # Take a step
                optimizer.step()

                # Take care of logging here
                nll_log[e_i] = nll.item()

            # Provide user with an update on progress here
            if e_i % update_int == 0:
                print('Epoch ' + str(e_i) + ' complete. NLL: ' + str(nll.item()))

        return nll_log

    def project(self, x: torch.Tensor, w: OptionalTensor = None, s_in: OptionalTensor = None,
                 b_in: OptionalTensor = None, apply_scales_and_biases: bool = True):
        """ Projects input data into the low-d space of the regression model.

        Args:

              x: Input data of shape n_smps*input_dim

              w: Optional weights to use in place of those of this object

              s_in: Optional s_in parameter to use in place of that of this object

              b_in: Optional s_in parameter to use in place of that of this object

              apply_scales_and_biases: True if input scales and biases should be applied.

        Returns:

             z: The project data of shape n_smps*p
         """
        if w is None:
            w = self.w
        if b_in is None:
            b_in = self.b_in
        if s_in is None:
            s_in = self.s_in

        if apply_scales_and_biases:
            return s_in*torch.matmul(x, w) + b_in
        else:
            return torch.matmul(x, w)

    def forward(self, x: torch.Tensor, w: OptionalTensor = None, s_in: OptionalTensor = None,
                b_in: OptionalTensor = None, s_out: OptionalTensor = None,
                b_out: OptionalTensor = None) -> torch.Tensor:
        """ Computes E(y|x) for the model.

        Args:

            x: Conditioning input to generate samples from. Of shape n_smps*n_input_variables

            w: If provided, this is used in place of the w_in parameter of the model

            s_in: If provided, this is used in place of the s_in parameter of the model

            b_in: If provided, this is used in place of the b_in parameter of the model

            s_out: If provided, this is used in place of the s_out parameter of the model

            b_out: If provided, this is used in place of the b_out parameter of the model

        Returns:

            y: E(y|x) of shape n_smps*n_predicted_variables

        """

        if w is None:
            w = self.w
        if b_in is None:
            b_in = self.b_in
        if s_in is None:
            s_in = self.s_in
        if s_out is None:
            s_out = self.s_out
        if b_out is None:
            b_out = self.b_out

        return self.cond_mean(x=x, w=w, s_in=s_in, b_in=b_in, s_out=s_out, b_out=b_out)

    def sample(self, x: torch.Tensor, w: OptionalTensor = None, s_in: OptionalTensor = None,
               b_in: OptionalTensor = None, s_out: OptionalTensor = None, b_out: OptionalTensor = None,
               psi: OptionalTensor = None) -> torch.Tensor:
        """ Generates samples from the model.

        Args:

            x: Conditioning input to generate samples from. Of shape n_smps*n_input_variables

            w: If provided, this is used in place of the w_in parameter of the model

            s_in: If provided, this is used in place of the s_in parameter of the model

            b_in: If provided, this is used in place of the b_in parameter of the model

            s_out: If provided, this is used in place of the s_out parameter of the model

            b_out: If provided, this is used in place of the b_out parameter of the model

            psi: If provided, this is used in place of the psi parameter of the model

        Returns:

            y: Samples of shape n_smps*n_predicted_variables

        """

        if psi is None:
            psi = self.psi

        mns = self(x=x, w=w, s_in=s_in, b_in=b_in, s_out=s_out, b_out=b_out)
        noise = torch.randn(mns.shape, device=psi.device)*torch.sqrt(psi)

        return mns + noise


def fit_with_hypercube_priors(data: Sequence[Sequence[torch.Tensor]], props: Sequence[torch.Tensor], p: int,
                              dense_net_opts: dict, sp_fit_opts: Sequence[dict], ip_fit_opts: Sequence[dict],
                              w_prior_opts: OptionalDict = None,  s_in_prior_opts: OptionalDict = None,
                              b_in_prior_opts: OptionalDict = None, s_out_prior_opts: OptionalDict = None,
                              b_out_prior_opts: OptionalDict = None, psi_prior_opts: OptionalDict = None,
                              w_post_opts: OptionalDict = None, s_in_post_opts: OptionalDict = None,
                              b_in_post_opts: OptionalDict = None, s_out_post_opts: OptionalDict = None,
                              b_out_post_opts: OptionalDict = None, psi_post_opts: OptionalDict = None,
                              sp_fixed_var: bool = False, fixed_s_in_vl: OptionalTensor = None,
                              fixed_b_in_vl: OptionalTensor = None, fixed_s_out_vl: OptionalTensor = None,
                              fixed_b_out_vl: OptionalTensor = None,
                              match_init_ip_w_post_to_prior: bool = True) -> dict:
    """ Wrapper function to setup models and fit them to data for multiple systems using hypercube priors over weights.

    Note: This function allows the user to chose whether to learn the s_in, b_in, s_out and b_out parameters for each
    system probabilistically or to assume these are fixed and the same for all systems.  If they should be learned
    probabilistically, full dictionaries of appropriate options should be provided to the appropriate prior and
    posterior inputs.  If they are fixed, appropriate tensors of fixed values should be provided to the appropriate
    "fixed" arguments (see below).

    Args:

        data: The data for each system (e.g. individual).  Each entry is of the form [input_vars, predicted_vars],
        where input_vars is of shape n_smp*n_input_vars and predicted variables is of shape n_smps*n_predicted_vars.

        props: The properties for each system.  Each entry is of shape n_input_vars*n_props

        p: The dimensionality of the intermediate low-d space we project input variables down into

        dense_net_opts: Dictionary of options for setting up the dense net of the m module in the regression models.
        Should have the entries, 'n_layers', 'growth_rate' and 'bias'.

        sp_fit_opts: sp_fit_opts[i] are the options to the call to fit for the i^th round of fitting shared-posterior
        models

        ip_fit_opts: ip_fit_opts[i] are the options to the call to fit for the i^th round of fitting
        individual-posterior models

        w_prior_opts: Options to provide to generate_hypercube_prior_collection for generating the prior on the weights

        s_in_prior_opts: A dictionary of options to provide to generate_hypercube_prior_collection for generating
        the prior on the s_in parameters.  If None, default options will be used.  If fixed_s_in_vl is not None, this
        will be ignored.

        b_in_prior_opts: A dictionary of options to provide to generate_hypercube_prior_collection for generating
        the prior on the b_in parameters.  If None, default options will be used. If fixed_b_in_vl is not None,
        this will be ignored.

        s_out_prior_opts: A dictionary of options to provide to generate_hypercube_prior_collection for
        generating the prior on the s_out parameters.  If None, default options will be used. If fixed_s_out_vl is not
        None, this will be ignored.

        b_out_prior_opts: A dictionary of options to provide to generate_hypercube_prior_collection for
        generating the prior on the b_out parameters. If None, default options will be used. If fixed_b_out_vl is not
        None, this will be ignored.

        psi_prior_opts: options to provide to generate_hypercube_prior_collection for generating the prior on
        the psi parameters. If None, default options will be used.

        w_post_opts: Options to provide to generate_basic_posteriros for generating posteriors of the weights.  If None,
        default values will be used.

        s_in_post_opts: Options to provide to generate_basic_posteriors for generating posteriors over
        the s_in parameters. If None, default options will be used. If fixed_s_in_vl is not None, this will be ignored.

        b_in_post_opts: Options to provide to generate_basic_posteriors for generating posteriors over
        the b_in parameters. If None, default options will be used. If fixed_b_in_vl is not None, this will be ignored.

        s_out_post_opts: Options to provide to generate_basic_posteriors for generating posteriors over
        the s_out parameters. If None, default options will be used. If fixed_s_out_vl is not None, this will be ignored.

        b_out_post_opts: Options to provide to generate_basic_posteriors for generating posteriors over
        the b_out parameters. If None, default options will be used. If fixed_b_out_vl is not None, this will be ignored.

        psi_post_opts: options to provide to generate_basic_posteriors for generating posteriors over
        the psi parameters. If None, default options will be used.

        sp_fixed_var: True if the variance of the shared posteriors in the sp fitting stage should be fixed to
        their initial value.

        fixed_b_in_vl: Fixed values that b_in should be set to for all systems.  If this argument is None, b_in
        will be treated as not fixed and instead a posterior distribution will be learned over these parameters for
        all systems.

        fixed_s_in_vl: Fixed values that s_in should be set to for all systems.  If this argument is None, s_in
        will be treated as not fixed and instead a posterior distribution will be learned over these parameters for
        all systems.

        fixed_s_out_vl: Fixed values that s_out should be set to for all systems.  If this argument is None, s_out
        will be treated as not fixed and instead a posterior distribution will be learned over these parameters for
        all systems.

        fixed_b_out_vl: Fixed values that b_out should be set to for all systems.  If this argument is None, b_out
        will be treated as not fixed and instead a posterior distribution will be learned over these parameters for
        all systems.

        match_init_ip_w_post_to_prior: True if the initial posterior over weights for inidividual-posterior training
        should be set equal to the initial CPD for each individual.

    Returns: Results of the fitting.  This will be a dictionary with entires 'sp' and 'ip' containing results of
    fitting the shared and individual posterior models respectively.  Each will itself be a dictionary with the
    keys 'vi_collections', 'priors' and 'logs' holding the fit vi collections, priors and fitting logs.

    """

    n_systems = len(data)
    n_pred_vars = data[0][1].shape[1]
    ind_n_vars = [d[0].shape[1] for d in data]
    #n_props = props[0].shape[1]

    # Determine if we are working with fixed scales and offsets
    fixed_s_in = fixed_s_in_vl is not None
    fixed_b_in = fixed_b_in_vl is not None
    fixed_s_out = fixed_s_out_vl is not None
    fixed_b_out = fixed_b_out_vl is not None
    print('fixed_s_in: ' + str(fixed_s_in))

    # ==================================================================================================================
    # See which devices are available for fitting
    # ==================================================================================================================
    devices, _ = list_torch_devices()

    # ==================================================================================================================
    # Create distributions and models for sp fitting
    # ==================================================================================================================

    # Setup the shared m-module
    sp_m = torch.nn.Sequential(DenseLNLNet(nl_class=torch.nn.ReLU,
                                           d_in=p,
                                           n_layers=dense_net_opts['n_layers'],
                                           growth_rate=dense_net_opts['growth_rate'],
                                           bias=dense_net_opts['bias']),
                               torch.nn.Linear(in_features=p+dense_net_opts['n_layers']*dense_net_opts['growth_rate'],
                                               out_features=n_pred_vars,
                                               bias=True))

    # Setup the priors
    sp_priors = generate_hypercube_prior_collection(p=p, d_pred=n_pred_vars, w_prior_opts=w_prior_opts,
                                                    s_in_prior_opts=s_in_prior_opts, b_in_prior_opts=b_in_prior_opts,
                                                    s_out_prior_opts=s_out_prior_opts,
                                                    b_out_prior_opts=b_out_prior_opts, psi_prior_opts=psi_prior_opts,
                                                    learnable_scales_and_biases=False, fixed_s_in=fixed_s_in,
                                                    fixed_b_in=fixed_b_in, fixed_s_out=fixed_s_out,
                                                    fixed_b_out=fixed_b_out)

    # Fixed the variance of the sp prior if we are suppose to
    if sp_fixed_var:
        print('Fixing variance of sp w_prior.')
        for d in sp_priors.w_prior.dists:
            for param in d.std_f.parameters():
                param.requires_grad = False

    # Setup the posteriors - we don't provide w_post opts here as individual posteriors are not used in sp fitting
    sp_posteriors = generate_basic_posteriors(n_input_vars=ind_n_vars, p=p, n_pred_vars=n_pred_vars,
                                              s_in_opts=s_in_post_opts, b_in_opts=b_in_post_opts,
                                              s_out_opts=s_out_post_opts, b_out_opts=b_out_post_opts,
                                              psi_opts=psi_post_opts, fixed_s_in=fixed_s_in, fixed_b_in=fixed_b_in,
                                              fixed_s_out=fixed_s_out, fixed_b_out=fixed_b_out)

    # Setup the model objects - handling fixed parameters
    sp_mdls = [GNLRMdl(m=sp_m, s_in=fixed_s_in_vl, b_in=fixed_b_in_vl, s_out=fixed_s_out_vl, b_out=fixed_b_out_vl)
               for s in range(n_systems)]

    for mdl in sp_mdls:
        for fixed, param in [(fixed_s_in, mdl.s_in), (fixed_b_in, mdl.b_in), (fixed_s_out, mdl.s_out),
                             (fixed_b_out, mdl.b_out)]:
            if fixed:
                param.requires_grad = False

    # ==================================================================================================================
    # Tie posteriors for weights together
    # ==================================================================================================================
    for posteriors in sp_posteriors:
        posteriors.w_post = sp_priors.w_prior

    # ==================================================================================================================
    # Setup the vi collections
    # ==================================================================================================================
    sp_vi_collections = [VICollection(data=[data[s][0], data[s][1]], props=props[s], mdl=sp_mdls[s],
                                      posteriors=sp_posteriors[s])
                         for s in range(n_systems)]

    # ==================================================================================================================
    # Perform sp fitting
    # ==================================================================================================================
    print('Beginning SP fitting.')
    sp_fitter = Fitter(vi_collections=sp_vi_collections, priors=sp_priors, devices=devices)

    sp_fitter.distribute(distribute_data=True, devices=devices)
    prev_sp_epochs = np.cumsum([0] + [fit_opts['n_epochs'] for fit_opts in sp_fit_opts])
    sp_logs = [sp_fitter.fit(skip_w_kl=True, prev_epochs=prev_epochs, **fit_opts)
               for prev_epochs, fit_opts in zip(prev_sp_epochs, sp_fit_opts)]
    sp_fitter.distribute(distribute_data=True, devices=[torch.device('cpu')])

    # ==================================================================================================================
    # Create distributions and models for ip fitting
    # ==================================================================================================================

    # We initialize the shared-m module for ip training to that we learned in sp training
    ip_m = copy.deepcopy(sp_m)

    # We initialize the priors for ip training to those we learned in sp training
    ip_priors = copy.deepcopy(sp_priors)

    # Set the variance of ip posteriors for weights to be learnable if we need to
    if sp_fixed_var:
        for d in ip_priors.w_prior.dists:
            for param in d.std_f.parameters():
                param.requires_grad = True

    # Create the posteriors for ip training, we will set many of the posteriors to copies of the sp posteriors later
    ip_posteriors = generate_basic_posteriors(n_input_vars=ind_n_vars, p=p, n_pred_vars=n_pred_vars,
                                              w_opts=w_post_opts, s_in_opts=s_in_post_opts, b_in_opts=b_in_post_opts,
                                              s_out_opts=s_out_post_opts, b_out_opts=b_out_post_opts,
                                              psi_opts=psi_post_opts, fixed_s_in=fixed_s_in, fixed_b_in=fixed_b_in,
                                              fixed_s_out=fixed_s_out, fixed_b_out=fixed_b_out)

    # Create model objects - handling fixed parameters

    ip_mdls = [GNLRMdl(m=ip_m, s_in=fixed_s_in_vl, b_in=fixed_b_in_vl, s_out=fixed_s_out_vl, b_out=fixed_b_out_vl)
               for _ in range(n_systems)]

    for mdl in ip_mdls:
        for fixed, param in [(fixed_s_in, mdl.s_in), (fixed_b_in, mdl.b_in), (fixed_s_out, mdl.s_out),
                             (fixed_b_out, mdl.b_out)]:
            if fixed:
                param.requires_grad = False

    # ==================================================================================================================
    # Initialize the posteriors for the ip models with sp solutions
    # ==================================================================================================================
    for s in range(n_systems):
        # We can just copy over distributions for scales and biases b/c these were fit individually in the sp fitting
        ip_posteriors[s] = PosteriorCollection(w_post=ip_posteriors[s].w_post,
                                               s_in_post=copy.deepcopy(sp_posteriors[s].s_in_post),
                                               b_in_post=copy.deepcopy(sp_posteriors[s].b_in_post),
                                               s_out_post=copy.deepcopy(sp_posteriors[s].s_out_post),
                                               b_out_post=copy.deepcopy(sp_posteriors[s].b_out_post),
                                               psi_post=copy.deepcopy(sp_posteriors[s].psi_post))

        # Here we make the initial posteriors for the weights equal to the initial prior for each individual
        if match_init_ip_w_post_to_prior:
            if 'dists' in dir(sp_priors.w_prior):
                with torch.no_grad():
                    for d_i in range(p):
                        cur_mn = sp_priors.w_prior.dists[d_i](props[s]).squeeze()
                        cur_std = sp_priors.w_prior.dists[d_i].std_f(props[s]).squeeze().numpy()
                        ip_posteriors[s].w_post.dists[d_i].mn_f.f.vl.data = copy.deepcopy(cur_mn)
                        ip_posteriors[s].w_post.dists[d_i].std_f.f.set_value(copy.deepcopy(cur_std))
            else:
                with torch.no_grad():
                    cur_mn = sp_priors.w_prior(props[s])
                    cur_std = sp_priors.w_prior.std_f(props[s])
                    for d_i in range(p):
                        ip_posteriors[s].w_post.dists[d_i].mn_f.f.vl.data = copy.deepcopy(cur_mn[:, d_i])
                        ip_posteriors[s].w_post.dists[d_i].std_f.f.set_value(copy.deepcopy(cur_std[:, d_i].squeeze().numpy()))

    # ==================================================================================================================
    # Setup the vi collections
    # ==================================================================================================================
    ip_vi_collections = [VICollection(data=[data[s][0], data[s][1]], props=props[s], mdl=ip_mdls[s],
                                      posteriors=ip_posteriors[s])
                         for s in range(n_systems)]

    # ==================================================================================================================
    # Perform ip fitting
    # ==================================================================================================================
    print('Beginning IP fitting.')
    ip_fitter = Fitter(vi_collections=ip_vi_collections, priors=ip_priors, devices=devices)

    ip_fitter.distribute(distribute_data=True, devices=devices)
    prev_ip_epochs = np.cumsum([0] + [fit_opts['n_epochs'] for fit_opts in ip_fit_opts])
    ip_logs = [ip_fitter.fit(**fit_opts, prev_epochs=prev_epochs)
               for prev_epochs, fit_opts in zip(prev_ip_epochs, ip_fit_opts)]

    ip_fitter.distribute(distribute_data=True, devices=[torch.device('cpu')])

    # ==================================================================================================================
    # Return output
    # ==================================================================================================================
    sp_rs = {'vi_collections': sp_vi_collections, 'priors': sp_priors, 'logs': sp_logs}
    ip_rs = {'vi_collections': ip_vi_collections, 'priors': ip_priors, 'logs': ip_logs}

    return {'sp': sp_rs, 'ip': ip_rs}


def generate_basic_posteriors(n_input_vars: Sequence[int], p: int, n_pred_vars: int, w_opts: OptionalDict = None,
                              s_in_opts: OptionalDict = None, b_in_opts: OptionalDict = None,
                              s_out_opts: OptionalDict = None, b_out_opts: OptionalDict = None,
                              psi_opts: OptionalDict = None, fixed_s_in: bool = False,
                              fixed_b_in: bool = False, fixed_s_out: bool = False,
                              fixed_b_out: bool = False):

    """ Generates basic posteriors over model parameters for a set of models.

    By basic posteriors, we mean the posteriors are not conditioned on properties but instead
    are represented by a product of distributions over each coefficient in the parameters, with
    the distribution for each coefficient being learned independently.

    We represent posterior distributions over the coefficients of the weights, s_in, b_in, s_out and b_out parameters
    as Gaussians and posterior distributions over noise variances as Gamma distribitions.

    Note: The user can optionally chose to not create posteriors for the scales and biases (as there may be certain
    scenarios where we assume these are known and fixed) by setting the appropriate function arguments (see below)
    to true.


    Args:

        n_input_vars: n_input_vars[i] is the number of input variables for model i

        p: The number of low-d variables input variables are projected down to

        n_pred_vars: The number of predicted variables

        w_opts: Options to provide to MatrixGaussianProductDistribtion when creating the posteriors over weights.
        All options except shape can be specified.

        s_in_opts: Options to provide to MatrixGaussianProductDistribtion when creating the posteriors over s_in
        parameters. All options except shape can be specified.

        b_in_opts: Options to provide to MatrixGaussianProductDistribtion when creating the posteriors over b_in
        parameters. All options except shape can be specified.

        s_out_opts: Options to provide to MatrixGaussianProductDistribtion when creating the posteriors over s_out
        parameters. All options except shape can be specified.

        b_out_opts: Options to provide to MatrixGaussianProductDistribtion when creating the posteriors over b_out
        parameters. All options except shape can be specified.

        psi_opts: Options to provide to MatrixGammaProductDistribution when creating posteriors over psi parameters.
        All options except shape can be specified.

    Returns:

        post_collections: post_colletions[i] contains the posterior collections for subject i.

    """

    if w_opts is None:
        w_opts = dict()
    if s_in_opts is None:
        s_in_opts = dict()
    if b_in_opts is None:
        b_in_opts = dict()
    if s_out_opts is None:
        s_out_opts = dict()
    if b_out_opts is None:
        b_out_opts = dict()
    if psi_opts is None:
        psi_opts = dict()

    n_mdls = len(n_input_vars)
    post_collections = [None]*n_mdls

    for i, n_i in enumerate(n_input_vars):
        w_post = MatrixGaussianProductDistribution(shape=[n_i, p], **w_opts)

        if not fixed_s_in:
            s_in_post = MatrixGaussianProductDistribution(shape=[p, 1], **s_in_opts)
        else:
            s_in_post = None

        if not fixed_b_in:
            b_in_post = MatrixGaussianProductDistribution(shape=[p, 1], **b_in_opts)
        else:
            b_in_post = None

        if not fixed_s_out:
            s_out_post = MatrixGaussianProductDistribution(shape=[n_pred_vars, 1], **s_out_opts)
        else:
            s_out_post = None

        if not fixed_b_out:
            b_out_post = MatrixGaussianProductDistribution(shape=[n_pred_vars, 1], **b_out_opts)
        else:
            b_out_post = None

        psi_post = MatrixGammaProductDistribution(shape=[n_pred_vars,1], **psi_opts)

        post_collections[i] = PosteriorCollection(w_post=w_post, s_in_post=s_in_post, b_in_post=b_in_post,
                                                  s_out_post=s_out_post, b_out_post=b_out_post, psi_post=psi_post)

    return post_collections


def generate_hypercube_prior_collection(p: int, d_pred: int, w_prior_opts: dict, s_in_prior_opts: Optional[dict] = None,
                                        b_in_prior_opts: Optional[dict] = None, s_out_prior_opts: Optional[dict] = None,
                                        b_out_prior_opts: Optional[dict] = None, psi_prior_opts: Optional[dict] = None,
                                        learnable_scales_and_biases: bool = True, fixed_s_in: bool = False,
                                        fixed_b_in: bool = False, fixed_s_out: bool = False,
                                        fixed_b_out: bool = False) -> 'PriorCollection':
    """ Generates conditional priors where sums of hypercube functions of properties predict mn and std of weights.

    For the remaining parameters of the model,

        1) Scales and biases are distributed according to Gaussians which are not conditioned on properties (more
        specifically, they ignore the conditioning property input)

        2) Psi parameters are distributed according to Gamma distributions which are not conditioned on input (again,
        they ignore the conditioning property input)

    The conditional prior over the weights will be Gaussian, with:

        1) Means which are a sum of hypercube functions

        2) Standard deviations which are passed through a sum of hypercube functions (of properties),
        and then passed through an exponential plus a fixed offset (to prevent standard deviations from
        going below a certain bound).

    Note: The user can optionally chose to not create priors for the input scales and biases (as there may be certain
    scenarios where we assume these are known and fixed) byt setting the appropriate function arguments (see below)
    to true.

    Args:
        p: The number of low-d variables input data is projected to

        d_pred: The number of predicted variables.

        w_prior_opts: Options to pass to CondMatrixHypercubePrior.  This must contain the keys, mn_hc_params,
        std_hc_params and min_std.  It should not contain n_cols, which will be determined by p.
        See CondMatrixHypercubePrior for more details.

        s_in_prior_opts: Options to pass MatrixGaussianProductDistribution when generating the prior for the s_in
        parameters.  All options can be passed except for shape, which is determined by p.

        b_in_prior_opts: Options to pass MatrixGaussianProductDistribution when generating the prior for the b_in
        parameters.  All options can be passed except for shape, which is determined by p.

        s_out_prior_opts: Options to pass MatrixGaussianProductDistribution when generating the prior for the s_out
        parameters.  All options can be passed except for shape, which is determined by d_pred.

        b_out_prior_opts: Options to pass MatrixGaussianProductDistribution when generating the prior for the b_out
        parameters.  All options can be passed except for shape, which is determined by d_pred.

        psi_prior_opts: Options to pass to MatrixGammaProductDistribution when generating the prior for the psi
        parameters.  All options can be passed except for shape, which is determined by d_pred.

        learnable_scales_and_biases: True if priors on scales and biases (both in and out) should be learnable.

        fixed_b_in: True if a prior for b_in parameters should not be created.

        fixed_s_in: True if a prior for s_in parameters should not be created.

        fixed_b_out: True if a prior for b_out parameters should not be created.

        fixed_s_out: True if a prior for s_out parameters should not be created.

    Returns:

        priors: The generated collection of priors.
    """

    if s_in_prior_opts is None:
        s_in_prior_opts = dict()
    if b_in_prior_opts is None:
        b_in_prior_opts = dict()
    if s_out_prior_opts is None:
        s_out_prior_opts = dict()
    if b_out_prior_opts is None:
        b_out_prior_opts = dict()
    if psi_prior_opts is None:
        psi_prior_opts = dict()

    # Generate prior for weights
    w_prior = CondMatrixHypercubePrior(n_cols=p, **w_prior_opts)

    # Generate prior for s_in
    if not fixed_s_in:
        s_in_prior = MatrixGaussianProductDistribution(shape=[p, 1], **s_in_prior_opts)
    else:
        s_in_prior = None

    # Generate prior for b_in
    if not fixed_b_in:
        b_in_prior = MatrixGaussianProductDistribution(shape=[p, 1], **b_in_prior_opts)
    else:
        b_in_prior = None

    # Generate prior for s_out
    if not fixed_s_out:
        s_out_prior = MatrixGaussianProductDistribution(shape=[d_pred, 1], **s_out_prior_opts)
    else:
        s_out_prior = None

    # Generate prior for b_out
    if not fixed_b_out:
        b_out_prior = MatrixGaussianProductDistribution(shape=[d_pred, 1], **b_out_prior_opts)
    else:
        b_out_prior = None

    # Generate prior for psi
    psi_prior = MatrixGammaProductDistribution(shape=[d_pred, 1], **psi_prior_opts)

    # Make it so that distributions for scales and biases have non-learnable parameters if we are suppose to
    if not learnable_scales_and_biases:
        for d in [s_in_prior, b_in_prior, s_out_prior, b_out_prior]:
            if d is not None:
                for p in d.parameters():
                    p.requires_grad = False

    return PriorCollection(w_prior=w_prior, s_in_prior=s_in_prior, b_in_prior=b_in_prior,
                           s_out_prior=s_out_prior, b_out_prior=b_out_prior, psi_prior=psi_prior)


def load_check_points(cp_dir: pathlib.Path, cp_str: str = None) -> Tuple[np.ndarray, List[dict]]:
    """ Finds and loads check points in a directory.

    The returned check points will be sorted by the training epoch they were saved after.

    Args:

        cp_dir: The directory that check points are saved in

        cp_str: The string at the start of file names denoting check points.  If None, 'cp_' will be used.

    Returns:

        cp_rs: The loaded check points.

        cp_epochs: The epochs each check point was saved after.

    """

    if cp_str is None:
        cp_str = 'cp_'

    # Load check points
    cp_files = glob.glob(str(cp_dir / (cp_str + '*.pt')))
    n_cps = len(cp_files)
    cp_rs = [None] * n_cps
    for cp_i, cp_file in enumerate(cp_files):
        cp_rs[cp_i] = torch.load(cp_file)
        print('Done loading check point ' + str(cp_i + 1) + ' of ' + str(n_cps) + '.')

    # Sort check points by epoch
    cp_epochs = np.asarray([cp['total_epoch'] for cp in cp_rs])
    cp_sort_order = np.argsort(cp_epochs)
    cp_epochs = cp_epochs[cp_sort_order]
    cp_rs = [cp_rs[i] for i in cp_sort_order]

    return cp_rs, cp_epochs


class PosteriorCollection():
    """ Contains posteriors over the parameters of an GNLR model.

    By holding all of the posteriors together, we gain convenience, when doing things such as moving the
    posteriors from one device to another or getting parameters.

    """

    def __init__(self, w_post: OptionalTensor = None, s_in_post: OptionalDistribution = None,
                 b_in_post: OptionalDistribution = None, s_out_post: OptionalDistribution = None,
                 b_out_post: OptionalDistribution = None, psi_post: OptionalDistribution = None):
        """ Creates a new PosteriorCollection.

        Note: A user does not need to specify posteriors for all types of parameters (as there may be certain scenarios
        where we treat some parameters as fixed).  To indicate that posteriors should not be generated for a certain
        type of parameter, simply set the appropriate argument to None.

        Args:

            w_post: The posterior over weights

            s_in_post: The posterior over the s_in parameter

            b_in_post: The posterior over the b_in parameter

            s_out_post: The posterior over the s_out parameter

            b_out_post: The posterior over the b_out parameter

            psi_post: The posterior over noise variances.

        """

        self.w_post = w_post
        self.s_in_post = s_in_post
        self.b_in_post = b_in_post
        self.s_out_post = s_out_post
        self.b_out_post = b_out_post
        self.psi_post = psi_post

    @staticmethod
    def from_checkpont(cp: dict) -> 'PosteriorCollection':
        """ Generates a new PosteriorCollection from a checkpoint dictionary.

        Args:

            cp: The checkpoint dictionary

        Returns:

            collection: The new collection
        """

        return PosteriorCollection(w_post=cp['w_post'], s_in_post=cp['s_in_post'], b_in_post=cp['b_in_post'],
                                   s_out_post=cp['s_out_post'], b_out_post=cp['b_out_post'], psi_post=cp['psi_post'])

    def generate_checkpoint(self):
        """ Generates a check point of the collection.

        The posteriors will be returned on cpu.

        Returns:
            cp: A dictionary with the keys 'w_post', 's_in_post', 'b_in_post', 's_out_post', 'b_out_post' and
            'psi_post' with values which will be the corresponding posteriors
        """

        cur_device = self.parameters()[0].device
        self.to('cpu')

        cp = {'w_post': copy.deepcopy(self.w_post),
              's_in_post': copy.deepcopy(self.s_in_post),
              'b_in_post': copy.deepcopy(self.b_in_post),
              's_out_post': copy.deepcopy(self.s_out_post),
              'b_out_post': copy.deepcopy(self.b_out_post),
              'psi_post': copy.deepcopy(self.psi_post)}

        self.to(cur_device)

        return cp

    def to(self, device: Union[str, torch.device]):
        """ Moves the collection the the specified device."""

        for post in self._all_dists():
            if post is not None:
                post.to(device)

    def parameters(self) -> List[torch.nn.Parameter]:
        """ Returns all parameters in all distributions in the collection. """

        return list(itertools.chain(*[post.parameters() for post in self._all_dists() if post is not None]))

    def _all_dists(self) -> list:
        return [self.w_post, self.s_in_post, self.b_in_post, self.s_out_post, self.b_out_post, self.psi_post]


def predict(coll: 'VICollection', x: torch.Tensor, sample: bool = False) -> torch.Tensor:
    """ Generates predictions from a model, given point estimates and distributions over model parameters.

    This function checks the model object in a VI Collection to see which parameters are estimated with point
    estimates.  For those that are not, it will use specific values based on the posterior distributions for
    these parameters.  These specific values can either be the mean of the distributions or samples from the
    distribution.

    Args:
         coll: The VICollection object containing the model and posterior distributions over parameters for the system
         we generate predictions for.

         x: Input data we use to generate predictions, of shape n_smps*n_input_variables.

         sample: If true, values for parameters represented as posterior distributions will be selected by sampling
         from these distributions.  If False, the means of the distributions will be used.

    Returns:

        preds: Predicted data, of shape n_smps*n_input_variables.
    """

    # Determine if there are any parameters for which point estimates are used in place of distributions
    w_point_estimate = coll.mdl.w is not None
    s_in_point_estimate = coll.mdl.s_in is not None
    b_in_point_estimate = coll.mdl.b_in is not None
    s_out_point_estimate = coll.mdl.s_out is not None
    b_out_point_estimate = coll.mdl.b_out is not None

    posteriors = coll.posteriors
    props = coll.props
    mdl = coll.mdl

    def _get_param_vl(post, props, point_estimate, squeeze):
        if not point_estimate:
            if sample:
                _, param_vl = _sample_posterior(post=post, props=props)
            else:
                param_vl = post(props)
            if squeeze:
                param_vl = param_vl.squeeze()
        else:
            # Passing None to appropriate functions will signal we use the parameter stored in the model object
            param_vl = None

        return param_vl

    # Get parameter values
    w = _get_param_vl(post=posteriors.w_post, props=props, point_estimate=w_point_estimate, squeeze=False)
    s_in = _get_param_vl(post=posteriors.s_in_post, props=props, point_estimate=s_in_point_estimate, squeeze=True)
    b_in = _get_param_vl(post=posteriors.b_in_post, props=props, point_estimate=b_in_point_estimate, squeeze=True)
    s_out = _get_param_vl(post=posteriors.s_out_post, props=props, point_estimate=s_out_point_estimate, squeeze=True)
    b_out = _get_param_vl(post=posteriors.b_out_post, props=props, point_estimate=b_out_point_estimate, squeeze=True)

    # Generate predictions
    return mdl.cond_mean(x=x, w=w, s_in=s_in, b_in=b_in, s_out=s_out, b_out=b_out)


class PriorCollection():
    """ Contains conditional priors over the parameters of an GNLR model.

    By holding all of the priors together, we gain convenience, when doing things such as moving the
    priors from one device to another to getting the parameters of priors.

    """

    def __init__(self, w_prior: OptionalDistribution = None, s_in_prior: OptionalDistribution = None,
                 b_in_prior: OptionalDistribution = None, s_out_prior: OptionalDistribution = None,
                 b_out_prior: OptionalDistribution = None, psi_prior: OptionalDistribution = None):
        """ Creates a new PriorCollection.

        Note: We do not need to specify all priors.  If we desire to omit a type of prior from the collection,
        simply set the appropriate parameter to None.

        Args:

            w_prior: The conditional prior over the regression weights.
            
            s_in_prior: The condition prior over the s_in parameter
            
            b_in_prior: The conditional prior over the b_in parameter
            
            s_out_prior: The conditional prior over the s_out parameter
            
            b_out_prior: The conditional prior over the b_out parameter

            psi_prior: The conditional prior over noise variances. 

        """
        self.w_prior = w_prior
        self.s_in_prior = s_in_prior
        self.b_in_prior = b_in_prior
        self.s_out_prior = s_out_prior
        self.b_out_prior = b_out_prior
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

    @staticmethod
    def from_checkpoint(cp: dict) -> 'PriorCollection':
        """ Generates a new PriorCollection from a checkpoint.

        Args:

            cp: The checkpoint dictionary

        Returns:

            coll: The new collection
        """

        return PriorCollection(w_prior=cp['w_prior'], s_in_prior=cp['s_in_prior'], b_in_prior=cp['b_in_prior'],
                               s_out_prior=cp['s_out_prior'], b_out_prior=cp['b_out_prior'], psi_prior=cp['psi_prior'])

    def generate_checkpoint(self):
        """ Generates a check point of the collection.

        The priors will be returned on cpu.

        Returns:
            cp: A dictionary with the keys 'w_prior', 's_in_prior', 'b_in_prior', 's_out_prior', 'b_out_prior' and
            'psi_prior' with values which will be corresponding priors
        """

        # First, see if there is anything to move
        move = False
        for prior in self._all_priors():
            if prior is not None:
                move = True

        if move:
            cur_device = self.parameters()[0].device
            self.to('cpu')

        cp = {'w_prior': copy.deepcopy(self.w_prior),
              's_in_prior': copy.deepcopy(self.s_in_prior),
              'b_in_prior': copy.deepcopy(self.b_in_prior),
              's_out_prior': copy.deepcopy(self.s_out_prior),
              'b_out_prior': copy.deepcopy(self.b_out_prior),
              'psi_prior': copy.deepcopy(self.psi_prior)}

        if move:
            self.to(cur_device)

        return cp

    def to(self, device: Union[str, torch.device]):
        """ Moves the collection the the specified device."""

        for prior in self._all_priors():
            if prior is not None:
                prior.to(device)

    def parameters(self) -> List[torch.nn.Parameter]:
        """ Returns all parameters present in all priors in the collection. """

        return list(itertools.chain(*[prior.parameters() for prior in self._all_priors() if prior is not None]))

    def _all_priors(self) -> List:
        return [self.w_prior, self.s_in_prior, self.b_in_prior, self.s_out_prior, self.b_out_prior, self.psi_prior]

class VICollection():
    """ A collection of objects necessary for fitting one GNLR model with variational inference.

    This is a wrapper object that contains everything needed for fitting one GNLR model: it contains the data
    the model is fit to, the properties associated with each of the observed random variables in the model,
    the posterior distribution over model parameters, and the GNLR model object itself.

    By placing all of these entities into a single object, certain things, like moving data and models between
    devices can be made more convenient.

     """

    def __init__(self, data: List[torch.Tensor], props: torch.Tensor, mdl: GNLRMdl,
                 posteriors: PosteriorCollection):
        """ Creates a new VICollection object.

        Args:

            data: The data the model is fit to.  data[0] contains input data of shape n_smps*n_input_vars and data[1]
            contains data to predict of shape n_smps*n_pred_vars

            props: The properties associated with each of the input variables in the model.
            Of shape n_input_vars*n_props

            mdl: The model object.  When creating this object, each parameter in the model for which posteriors
            will be fitted (instead of point estimates) should be set to None.

            posteriors: The collection of posteriors that will be fit over model parameters.

        """

        self.data = data
        self.props = props
        self.mdl = mdl
        self.posteriors = posteriors

    @staticmethod
    def from_checkpoint(cp: dict, data: Optional[List[torch.Tensor]] = None,
                        props: Optional[torch.Tensor] = None) -> 'VICollection':
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
            if self.data is not None:
                self.data[0] = self.data[0].to(device)
                self.data[1] = self.data[1].to(device)

        if self.props is not None:
            self.props = self.props.to(device)

        self.mdl.to(device)
        self.posteriors.to(device)

    def parameters(self) -> List[torch.nn.Parameter]:
        """ Returns parameters in the FA model and all posteriors in the collection. """

        return list(self.mdl.parameters()) + self.posteriors.parameters()

