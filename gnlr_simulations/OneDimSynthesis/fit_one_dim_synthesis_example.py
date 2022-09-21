""" A script to synthesize regression models in a simulated scenario.

In this scenario we:

    1) Generate random example systems and data from them as follows:

        a) Each example system is made up of "neurons" which are positioned randomly in a 2-d property space (their
           positions in this space are sampled from the uniform distribution over the unit square).  The number of
           neurons for each example system is also sampled uniformly from a given range.

        b) Each example system works by projecting neural data down into a 1-d space, running that projected value
        through a non-linear function (which is the same across all example systems) and then adding noise to
        create recorded behavior.

        c) The shared non-linear function is a sin function plus a constant ramping function.

        d) The weight of a neuron in projecting from neural data to the 1-d space is sampled from a conditional
        distribution, which is Gaussian conditioned on a neurons properties.  The mean and standard deviation of
        this conditional distribution are complicated functions of neuron positions, to make them difficult to learn.
        You can think of this distribution as specifying how the world generates example systems.

        e) The standard deviation of recording noise is also sampled idd from a Gamma distribution for each example
        system.

        f) We generate data for each example system to simulate recording limited patterns of neural activity from
        each.  In particular, we:

            i) Pick only a subset of x variables to be active (those in a contiguous region of property space) and
               designate the remaining x variables as silent

            ii) We generate data for the active x variables that project into a limited range of the domain of the
            shared non-linear function (simulating observing neural activity that drives a limited range of behavior
            from each example system).

    2) We then apply DPMS, using data form all example systems, to learn the CPD relating neuron weights to their
    position in propety space and posteriors over weights and noise standard deviations for each example system. When
    doing this, we learn mean and standard deviation functions for the CPD that are sum of hypercube basis functions
    (so these can not perfectly represent the true CPD in 1d) and we learn to represnt the shared non-linear function
    with a deep neural network (which again can not perfectly represent the true shared function)

    3) In addition, we also apply the DPMS framework to only one example system at a time.

    4) Finally, we save results for later plotting and analysis.

"""

import random

import numpy as np
import os
import pathlib
import torch

from janelia_core.ml.torch_distributions import MatrixGammaProductDistribution

from probabilistic_model_synthesis.gaussian_nonlinear_regression import fit_with_hypercube_priors
from probabilistic_model_synthesis.gaussian_nonlinear_regression import GNLRMdl
from probabilistic_model_synthesis.gaussian_nonlinear_regression import PriorCollection
from probabilistic_model_synthesis.simulation import generate_sum_of_bump_fcns_dist
from probabilistic_model_synthesis.simulation import efficient_cone_and_projected_interval_sample
from probabilistic_model_synthesis.simulation import IncreasingSinFcn
from probabilistic_model_synthesis.utilities import print_heading
from probabilistic_model_synthesis.utilities import print_info


# ======================================================================================================================
#  Parameters go here
# ======================================================================================================================
ps = dict()

# High-level switch determing what type of simulation we want to run - with or without props.  This only affects the way we
# set the rest of the parameters below and nothing else. 
ps['sim_type'] = 'props' #props or no_props

# The number of example systems we generate data for, must be >= 4
ps['n_ex_systems'] = 100  # Use >= 100 for publication

# Range of the number of input variables we observe from each example system - the actual number of variables we
# observe from an example system will be pulled uniformly from this range (inclusive)
ps['n_input_var_range'] = [10000, 11000]

# Range of the number of samples we observe from each example system - the actual number we observe from each
# example system will be unformly from this range (inclusive)

ps['n_smps_range'] = [7500, 9000] 

# Number of intermediate variables we project down into
ps['p'] = 1

# Number of variabales we predict
ps['d_pred'] = 1

# ===============================================================================================
# Parameters for the true priors

# Options for the prior distribution on weights
ps['true_w_prior_opts'] = {'n_bump_fcns': 50, 'd_in': 2, 'p': 1, 'mn_m_std': 1.0, 'std_m_std': .1, 'bump_w': .2}

# Options for the prior on noise standard deviations
ps['true_psi_prior_opts'] = {'conc_iv': 10.0, 'rate_iv': 1000.0, 'rate_lb': .1, 'rate_ub': 10000.0}

# ===============================================================================================
# Options for fixed scales and offsets (we don't learn these)

# Scales and offsets to apply when projecting down to the intermediate variables
ps['s_in'] = torch.tensor([1.0 / np.sqrt(ps['n_input_var_range'][0])] * ps['p'], dtype=torch.float)
ps['b_in'] = torch.tensor([0.0] * ps['p'], dtype=torch.float)

# Scales and offsets to apply after the m-module
ps['s_out'] = torch.tensor([1.0] * ps['d_pred'], dtype=torch.float)
ps['b_out'] = torch.tensor([0.0] * ps['d_pred'], dtype=torch.float)

# ===============================================================================================
# Parameters for setting up how we fit things with DPMS

if ps['sim_type'] == 'props':
    #The full options for setting up the prior on weights for limited data example
    fit_hc_params = {'n_divisions_per_dim': [100, 100],
                     'dim_ranges': np.asarray([[-.0001, 1.0001],
                                               [-.0001, 1.0001]]),
                     'n_div_per_hc_side_per_dim': [2, 2]}
elif ps['sim_type'] == 'no_props':
    # The full options for setupping up the prior on weights for the props vs no props example 
    # Essentially, we treat all properties the same with these settings 
    fit_hc_params = {'n_divisions_per_dim': [1, 1],
                     'dim_ranges': np.asarray([[-.0001, 1.0001],
                                               [-.0001, 1.0001]]),
                     'n_div_per_hc_side_per_dim': [1, 1]}

if ps['sim_type'] == 'props':
    # The full options for setting up the prior on weights when we have a CPD that can learn properties through space 
    ps['fit_w_prior_opts'] = {'mn_hc_params': fit_hc_params, 'std_hc_params': fit_hc_params,
                              'min_std': .000001, 'mn_init': 0.0, 'std_init': .01} #.3
elif ps['sim_type'] == 'no_props':
    # The full options for setting up the prior on weights when we have a CPD that ignore properties
    ps['fit_w_prior_opts'] = {'mn_hc_params': fit_hc_params, 'std_hc_params': fit_hc_params,
                              'min_std': .000001, 'mn_init': 0.0, 'std_init': .01}

# Options for prior on noise standard deviation
ps['fit_psi_prior_opts'] = ps['true_psi_prior_opts']

# Options for posterior distribtions
ps['psi_post_opts'] = {'conc_iv': 10.0, 'rate_iv': 1.0, 'rate_lb': .1, 'rate_ub': 10000.0}

# Options for the densenet which makes up the shared-m module
ps['dense_net_opts'] = {'n_layers': 2, 'growth_rate': 10, 'bias': True}

# ======================================================================================================
# Specify the example systems we fit in isolation
ps['single_fit_inds'] = range(0, ps['n_ex_systems'])

# ======================================================================================================
# Parameters for fitting - should be entered as lists, each entry corresponding to one round of fitting

if ps['sim_type'] == 'props':
    # Setting for when we have a CPD that can learn properties through space 
    ps['comb_sp_fit_opts'] = [{'n_epochs': 500, 'milestones': None, 'update_int': 100, 'init_lr': .01, 'n_batches': 2}]

    ps['comb_ip_fit_opts'] = [{'n_epochs': 1000, 'milestones': [1000], 'update_int': 100, 'init_lr': .1, 'n_batches': 2},
                              {'n_epochs': 1000, 'milestones': [1000], 'update_int': 100, 'init_lr': .01, 'n_batches': 2},
                              {'n_epochs': 1000, 'milestones': [1000], 'update_int': 100, 'init_lr': .001, 'n_batches': 2}]
elif ps['sim_type'] == 'no_props': 
    # Setting for when we have a CPD that ignore properties - in this case the shared prior initialiation won't help tie models together 
    ps['comb_sp_fit_opts'] = [{'n_epochs': 1, 'milestones': None, 'update_int': 100, 'init_lr': .0, 'n_batches': 2}]

    ps['comb_ip_fit_opts'] = [{'n_epochs': 1000, 'milestones': [1000], 'update_int': 100, 'init_lr': .001, 'n_batches': 2},
                              {'n_epochs': 1000, 'milestones': [1000], 'update_int': 100, 'init_lr': .001, 'n_batches': 2},
                              {'n_epochs': 1000, 'milestones': [1000], 'update_int': 100, 'init_lr': .001, 'n_batches': 2}]

ps['single_sp_fit_opts'] = ps['comb_sp_fit_opts']
ps['single_ip_fit_opts'] = ps['comb_ip_fit_opts']

# Specify the interval (in epochs) that check points should be created at
ps['cp_int'] = 100

# ======================================================================================================
# Determine if we set a random seed.  Can use this for reproducibility.

ps['random_seed'] = 1  # Set to None, to not set random seeds

# ======================================================================================================
# Location we should save results
ps['save_folder'] = r'/groups/bishop/bishoplab/projects/probabilistic_model_synthesis/results/simulation/gnlr/prototype_publication_results'
ps['save_file'] = 'with_props.pt'

# ======================================================================================================================
# Define helper functions here
# ======================================================================================================================


def _add_cps(fit_opts: dict, cp_folder: pathlib.Path, cp_save_str: str):
    """ Adds check point options to exiting fit options and creates folders to save check points in."""

    # Create directory to save check points in
    if not os.path.exists(cp_folder):
        os.makedirs(cp_folder)

    # Modify fit options so that check points are created
    for d in fit_opts:
        d['cp_epochs'] = list(range(0, d['n_epochs'], ps['cp_int'])) + [d['n_epochs']-1]
        d['cp_save_folder'] = cp_folder
        d['cp_save_str'] = cp_save_str


# ======================================================================================================================
#  See random seeds for reproducibility of results
# ======================================================================================================================
if ps['random_seed'] is not None:
    torch.manual_seed(ps['random_seed'])
    random.seed(ps['random_seed'])
    np.random.seed(ps['random_seed'])

# ======================================================================================================================
#  Create true distributions that govern how true systems are generated
# ======================================================================================================================
true_priors = PriorCollection(w_prior=generate_sum_of_bump_fcns_dist(**ps['true_w_prior_opts']),
                              psi_prior=MatrixGammaProductDistribution(shape=[ps['d_pred'], 1],
                                                                       **ps['true_psi_prior_opts']))

# ======================================================================================================================
# Define the true non-linear function relating projections of input variables to the mean of output variables
# ======================================================================================================================
m_true = IncreasingSinFcn()

# ======================================================================================================================
# Generate each example system and data
# ======================================================================================================================
print_heading('Generating synthetic systems and data.')

# Generate properties
ind_n_vars = np.random.randint(ps['n_input_var_range'][0], ps['n_input_var_range'][1]+1, ps['n_ex_systems'])
ind_props = [torch.rand(size=[n_vars,2]) for n_vars in ind_n_vars]

# Generate true models for each individual
with torch.no_grad():
    ind_true_mdls = [GNLRMdl(m=m_true,
                             w=true_priors.w_prior.form_standard_sample(true_priors.w_prior.sample(props)),
                             s_in=ps['s_in'],
                             b_in=ps['b_in'],
                             s_out=ps['s_out'],
                             b_out=ps['b_out'],
                             psi=true_priors.psi_prior.form_standard_sample(
                                 true_priors.psi_prior.sample(props)).squeeze(axis=1))
                     for props in ind_props]

    if ps['d_pred'] > 1:
        for mdl in ind_true_mdls:
            mdl.psi.data = mdl.psi.data.squeeze()

# Determine the property range of active neurons for each example system
ang_ranges = np.ones(ps['n_ex_systems'])
ang_ranges[0] = 0
ang_ranges *= .5*np.pi  # TODO: Should parameterize this
ang_ranges = np.cumsum(ang_ranges)
ang_ranges = [[a, a+np.pi] for a in ang_ranges]

# Generate data from each example system
ind_n_smps = np.random.randint(ps['n_smps_range'][0], ps['n_smps_range'][1]+1, ps['n_ex_systems'])
ind_data = [None]*ps['n_ex_systems']

min_proj_vl = -2*np.sqrt(ps['n_input_var_range'][0])
max_proj_vl = 2*np.sqrt(ps['n_input_var_range'][0])
interval_span = (max_proj_vl - min_proj_vl)/4

ind_intervals = [None]*ps['n_ex_systems']
for i in range(ps['n_ex_systems']):

    # Pick the interval projected data falls into
    if i < 4:
        start_interval = min_proj_vl + interval_span*i
    else:
        start_interval = np.random.uniform(low=min_proj_vl, high=max_proj_vl-interval_span)

    stop_interval = start_interval + interval_span

    cur_interval = [start_interval, stop_interval]
    ind_intervals[i] = cur_interval

    # Generate x data
    x_i_new = efficient_cone_and_projected_interval_sample(n_smps=ind_n_smps[i],
                                                           locs=ind_props[i],
                                                           ctr=torch.tensor([.5, .5]),
                                                           ang_range=ang_ranges[i],
                                                           w=ind_true_mdls[i].w.detach(),
                                                           interval=cur_interval,
                                                           big_std=1.0,
                                                           small_std=0,
                                                           device=torch.device('cuda'))
    x_i_new = x_i_new.cpu()

    with torch.no_grad():
        y_i_new = ind_true_mdls[i].sample(x=x_i_new)
    ind_data[i] = (x_i_new, y_i_new)

# ======================================================================================================================
# Apply DPMS to all example systems at once
# ======================================================================================================================
print_heading('Using DPMS to synthesize models using recorded data from all example systems.')

# Set things up for saving check points
comb_cp_folder = pathlib.Path(ps['save_folder']) / 'comb_cps'
_add_cps(fit_opts=ps['comb_sp_fit_opts'], cp_folder=comb_cp_folder, cp_save_str='sp_')
_add_cps(fit_opts=ps['comb_ip_fit_opts'], cp_folder=comb_cp_folder, cp_save_str='ip_')

comb_fit_rs = fit_with_hypercube_priors(data=ind_data, props=ind_props, p=ps['p'],
                                        dense_net_opts=ps['dense_net_opts'],
                                        sp_fit_opts=ps['comb_sp_fit_opts'],
                                        ip_fit_opts=ps['comb_ip_fit_opts'],
                                        w_prior_opts=ps['fit_w_prior_opts'],
                                        psi_prior_opts=ps['fit_psi_prior_opts'],
                                        psi_post_opts=ps['psi_post_opts'],
                                        sp_fixed_var=True,
                                        fixed_s_in_vl=ps['s_in'],
                                        fixed_b_in_vl=ps['b_in'],
                                        fixed_s_out_vl=ps['s_out'],
                                        fixed_b_out_vl=ps['b_out'])

# ======================================================================================================================
# Apply DPMS to a single example system
# ======================================================================================================================
print_heading('Fitting models to example systems in isolation.')

single_fit_rs = [None]*len(ps['single_fit_inds'])
for i, s_i in enumerate(ps['single_fit_inds']):
    print_info('Fitting model to example system ' + str(s_i) + '.')

    single_cp_folder = pathlib.Path(ps['save_folder']) / 'single_cps' / ('s_' + str(s_i))
    _add_cps(fit_opts=ps['single_sp_fit_opts'], cp_folder=single_cp_folder, cp_save_str='sp_')
    _add_cps(fit_opts=ps['single_ip_fit_opts'], cp_folder=single_cp_folder, cp_save_str='ip_')

    single_fit_rs[i] = fit_with_hypercube_priors(data=[ind_data[s_i]],
                                          props=[ind_props[s_i]], p=ps['p'],
                                          dense_net_opts=ps['dense_net_opts'],
                                          sp_fit_opts=ps['single_sp_fit_opts'],
                                          ip_fit_opts=ps['single_ip_fit_opts'],
                                          w_prior_opts=ps['fit_w_prior_opts'],
                                          psi_prior_opts=ps['fit_psi_prior_opts'],
                                          psi_post_opts=ps['psi_post_opts'],
                                          sp_fixed_var=True,
                                          fixed_s_in_vl=ps['s_in'],
                                          fixed_b_in_vl=ps['b_in'],
                                          fixed_s_out_vl=ps['s_out'],
                                          fixed_b_out_vl=ps['b_out'])

# ======================================================================================================================
# Save results
# ======================================================================================================================
print_heading('Saving results.')

rs = dict()
rs['ps'] = ps
rs['true_priors'] = true_priors
rs['m_true'] = m_true
rs['ind_n_vars'] = ind_n_vars
rs['ind_props'] = ind_props
rs['ang_ranges'] = ang_ranges
rs['ind_intervals'] = ind_intervals
rs['ind_data'] = ind_data
rs['ind_true_mdls'] = ind_true_mdls
rs['comb_fit_rs'] = comb_fit_rs
rs['single_fit_rs'] = single_fit_rs

torch.save(rs, pathlib.Path(ps['save_folder']) / ps['save_file'])

