""" For generating and saving parameters for the script syn_ahrens_gnlr_mdls.py. """

import copy
import pathlib
import pickle
import torch

import numpy as np

ps = dict()

# ======================================================================================================================
# Specify a note we want to save with the parameters (to summerize/remind a user what the particular settings are for)
# ======================================================================================================================
ps['note'] = ('Publication results.')

# ======================================================================================================================
#   Specify where these parameters are saved
# ======================================================================================================================

# Name of file parameters will be saved in
ps['param_filename'] = 'fit_params.pkl'

# Directory where we should save these parameters
ps['param_save_dir'] = r'/groups/fitzgerald/fitzgeraldlab/bishoplab/projects/probabilistic_model_synthsis/results/publication_results/gnlr/real_data'

# ======================================================================================================================
#   Specify where results will be saved
# ======================================================================================================================

ps['results_dir'] = None
ps['save_file'] = None

# ======================================================================================================================
#   Specify where check points should be saved for shared and individual posterior models - if not using check points
#   specify None.
# ======================================================================================================================
ps['sp_cp_dir'] = None
ps['ip_cp_dir'] = None

# ======================================================================================================================
# Specify which data we fit on and preprocessing options
# ======================================================================================================================

# Folder with the datasets in it
ps['data_dir'] = r'/groups/fitzgerald/fitzgeraldlab/bishoplab/projects/probabilistic_model_synthsis/data'

# Folder with the segment table in it
ps['segment_table_dir'] = r'/groups/fitzgerald/fitzgeraldlab/bishoplab/projects/probabilistic_model_synthsis/results/publication_results/fold_structures'

# File with the segment table in it
ps['segment_table_file'] = r'phototaxis_ns_subjects_1_2_5_6_8_9_10_11.pkl'

# Folder with the fold structure in it
ps['fold_str_dir'] = r'/groups/fitzgerald/fitzgeraldlab/bishoplab/projects/probabilistic_model_synthsis/results/publication_results/fold_structures'

# Give the name of the file with the fold structure in it
ps['fold_str_file'] = None

# Specify which fold we are fitting
ps['fold'] = None

# ======================================================================================================================
# Specify preprocessing options for the data
# ======================================================================================================================

# Indices of the behavioral variables we keep in the datasets
ps['keep_beh_vars'] = [3, 4]

# True if we should normalize the behavioral variables that we keep
ps['normalize_beh_vars'] = True

# Gains we apply to neural signals and behavioral variables
ps['neural_gain'] = 10000
ps['beh_gain'] = 100

# Ratio of z-plane spacing to voxel side length in x-y dimensions.
ps['z_ratio'] = 2.5

# Range of dimensions that ROIs fall in - used for setting up support for distributions
ps['roi_dim_ranges'] = np.asarray([[0, 1407],
                                   [0, 622],
                                   [0, np.ceil(137 * ps['z_ratio'])]])

# ======================================================================================================================
# Parameters for model structure
# ======================================================================================================================

ps['mdl_opts'] = dict()

# Dimensionality of intermediate low-d space
ps['mdl_opts']['p'] = 10

# Options for for generating the prior on the weights
hc_params = {'n_divisions_per_dim': [140, 50, 20],
             'dim_ranges': ps['roi_dim_ranges'],
             'n_div_per_hc_side_per_dim': [1, 1, 1]}


ps['mdl_opts']['w_prior_opts'] = {'mn_hc_params': hc_params,
                                  'std_hc_params': hc_params,
                                  'min_std': .000001,
                                  'mn_init': 0.0,
                                  'std_init': .01}

# Options for priors on input and output scales and biases.
#
# There are two main options here. We can fix the scales and offsets across all models, so they are not treated
# probabilistically.  In that case, the values they should be fixed to are specified by
# the 'fixed_<variable_name>_vl' dictionary entries below.
#
# However, if we want to allow the scales and offsets to vary across example systems, we should set
# the 'fixed_<variable_name>_vl' dictionary entries to None and instead provide values for the
# '<variable_name>_prior_opts' dictionary entries below to give the options for setting up the fixed priors over these.
# If the 'fixed_<variable_name>_vl' entries are not None, the options for setting up the priors will be ignored.

ps['mdl_opts']['fixed_s_in_vl'] = torch.tensor([.001]*ps['mdl_opts']['p'])
ps['mdl_opts']['fixed_b_in_vl'] = torch.tensor([0.0]*ps['mdl_opts']['p'])
ps['mdl_opts']['fixed_s_out_vl'] = torch.tensor([1.0]*len(ps['keep_beh_vars']))
ps['mdl_opts']['fixed_b_out_vl'] = torch.tensor([0.0]*len(ps['keep_beh_vars']))

ps['mdl_opts']['s_in_prior_opts'] = {'mn_mn': .001, 'mn_std': 1E-8, 'std_lb': 1E-8, 'std_ub': 1E-5, 'std_iv': 1E-6}
ps['mdl_opts']['b_in_prior_opts'] = {'mn_mn': 0.0, 'mn_std': 1E-8, 'std_lb': 1E-8, 'std_ub': 1E-5, 'std_iv': 1E-6}
ps['mdl_opts']['s_out_prior_opts'] = {'mn_mn': 1.0, 'mn_std': 1E-8, 'std_lb': 1E-6, 'std_ub': 1E-0, 'std_iv': 1E-4}
ps['mdl_opts']['b_out_prior_opts'] = {'mn_mn': 0.0, 'mn_std': 1E-8, 'std_lb': 1E-6, 'std_ub': 1E-0, 'std_iv': 1E-4}

# Options for prior on noise variances
ps['mdl_opts']['psi_prior_opts'] = {'conc_iv': 10.0,
                                    'rate_iv': 10.0,
                                    'conc_lb': 1.0,
                                    'conc_ub': 1000.0,
                                    'rate_lb': .1,
                                    'rate_ub': 10000.0,
                                    }

# Options for posterior on weights
ps['mdl_opts']['w_post_opts'] = {'std_lb': .000001,
                                 'std_ub': 10.0,
                                 'mn_mn': 0.0,
                                 'mn_std': .01,
                                 'std_iv': .01}

# Options for posteriors on input and output scales and biases.  Note that these will not be used if
# 'fixed_scales_and_offsets' above if false.
ps['mdl_opts']['s_in_post_opts'] = copy.deepcopy(ps['mdl_opts']['s_in_prior_opts'])
ps['mdl_opts']['b_in_post_opts'] = copy.deepcopy(ps['mdl_opts']['b_in_prior_opts'])
ps['mdl_opts']['s_out_post_opts'] = copy.deepcopy(ps['mdl_opts']['s_out_prior_opts'])
ps['mdl_opts']['b_out_post_opts'] = copy.deepcopy(ps['mdl_opts']['b_out_prior_opts'])

# Options for posterior on noise variances
ps['mdl_opts']['psi_post_opts'] = copy.deepcopy(ps['mdl_opts']['psi_prior_opts'])

# Options for the dense net m-module
ps['mdl_opts']['dense_net_opts'] = {'n_layers': 2, 'growth_rate': 10, 'bias': True}

# Option specifying if we fix the variance of the priors when doing sp fitting
ps['mdl_opts']['sp_fixed_var'] = True

# Options for fitting calls when fitting shared posterior and individual posterior models - all options accepted by
# the fit function can be provided here *except* the folders to save check points in, which are specified above

ps['mdl_opts']['sp_fit_opts'] = [{'n_epochs': 1000, 'n_batches': 2, 'init_lr': .0001, 'milestones': [500], 'gamma': .1,
                                  'update_int': 100, 'cp_epochs': list(range(0, 1000, 500)) + [999]} for _ in range(1)]

ps['mdl_opts']['ip_fit_opts'] = [{'n_epochs': 20000, 'n_batches': 2, 'init_lr': .00001, 'milestones': [10000], 'gamma': .1,
                                  'update_int': 100, 'cp_epochs': list(range(0, 20000, 500)) + [19999]} for _ in range(1)]

# ======================================================================================================================
# Save the parameters
# ======================================================================================================================

save_path = pathlib.Path(ps['param_save_dir']) / ps['param_filename']
with open(save_path, 'wb') as f:
    pickle.dump(ps, f)

print('Parameters saved to: ' + str(save_path))
