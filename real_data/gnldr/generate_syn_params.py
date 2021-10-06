""" For generating and saving parameters for the script syn_ahrens_gnlr_mdls.py. """

import copy
import pathlib
import pickle

import numpy as np

ps = dict()

# ======================================================================================================================
# Specify a note we want to save with the parameters (to summerize/remind a user what the particular settings are for)
# ======================================================================================================================
ps['note'] = 'Initial testing.'

# ======================================================================================================================
#   Specify where these parameters are saved
# ======================================================================================================================

# Name of file parameters will be saved in
ps['param_filename'] = 'transfer_params.pkl'

# Directory where we should save these parameters
ps['param_save_dir'] = r'/groups/bishop/bishoplab/projects/probabilistic_model_synthesis/results/real_data/gnldr/across_cond_transfer_analysis/v0'
#ps['param_save_dir'] = r'/groups/bishop/bishoplab/projects/probabilistic_model_synthesis/results/real_data/gnldr/same_cond_transfer_analysis/v2'

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
ps['data_dir'] = r'/groups/bishop/bishoplab/projects/ahrens_wbo/data'

# Folder with the segment table in it
ps['segment_table_dir'] = r'/groups/bishop/bishoplab/projects/probabilistic_model_synthesis/results/real_data'

# File with the segment table in it
#ps['segment_table_file'] = r'phototaxis_ns_subjects_1_2_5_6_8_9_10_11.pkl'
ps['segment_table_file'] = r'omr_l_r_f_ns_across_cond_segments_8_9_10_11.pkl'

# Folder with the fold structure in it
ps['fold_str_dir'] = r'/groups/bishop/bishoplab/projects/probabilistic_model_synthesis/results/real_data'

# Give the name of the file with the fold structure in it
ps['fold_str_file'] = None

# Specify which fold we are fitting
ps['fold'] = None

# ======================================================================================================================
# Specify preprocessing options for the data
# ======================================================================================================================

# Gains we apply to neural signals and behavioral variables
ps['neural_gain'] = 10

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

# Number of latent variables in the FA models
ps['mdl_opts']['n_latent_vars'] = 10

# Options for for generating the prior on the weights
hc_params = {'n_divisions_per_dim': [70, 25, 10],
             'dim_ranges': ps['roi_dim_ranges'],
             'n_div_per_hc_side_per_dim': [1, 1, 1]}

# Options for setting up priors
ps['mdl_opts']['prior_opts'] = {
                                # Parameters of the spatial grid underlying the priors
                                'hc_params': hc_params,
                                # Floor on std for distributions on means and loading matrices
                                'min_gaussian_std': 1E-5,  # Used to be .01,
                                # Floor on concentration parameter for any gamma distribution
                                'min_gamma_conc_vl': 1.0,
                                # Floor on rate parameter for any gamma distribution
                                'min_gamma_rate_vl': 1E-5,  # Used to be .01
                                # Mean of distribution used to initialize loading matrix
                                'lm_mn_init': 0.0,
                                # Std of distribution used to initialize loading matrix
                                'lm_std_init': .1,
                                # Mean of distribution used to initialize mean vectors
                                'mn_mn_init': 0.0,
                                # Std of distribution used to initialize mean vectors
                                'mn_std_init': .1,
                                # Initial value of concentration parameter for distributions over priavate variances
                                'psi_conc_vl_init': 10.0,
                                # Initial value of rate parameter for distsributions over private vairances
                                'psi_rate_vl_init': 10.0,
                                # False if standard deviations of distributions should be fixed (non-learnable)
                                'learnable_stds': True,
                                }

# Options for setting up posteriors
ps['mdl_opts']['post_opts'] = {
                              # Options for posteriors on loading matrices
                              'lm_opts': {'mn_mn': 1.0,
                                          'mn_std': 1E-8,
                                          'std_lb': 1E-5,
                                          'std_ub': 10.0,
                                          'std_iv': 1E-4},
                              # Options for posteriors on mean vectors
                              'mn_opts': {'mn_mn': 1.0,
                                          'mn_std': 1E-8,
                                          'std_lb': 1E-5,
                                          'std_ub': 10.0,
                                          'std_iv': 1E-4},
                              'psi_opts': {'alpha_lb': 1.0,
                                           'alpha_iv': 10.0,  # Used to be 5
                                           'beta_lb': 1E-5,
                                           'beta_iv': 10.0},  # Used to be 5
                             }

# Options for fitting shared posterior models
ps['mdl_opts']['sp_fit_opts'] = [{'n_epochs': 1000, 'milestones': [500], 'update_int': 100, 'init_lr': .1,
                                  'cp_epochs': list(range(0, 1000, 100)) + [999]} for _ in range(1)]

# Options for fitting individual posterior models
ps['mdl_opts']['ip_fit_opts'] = [{'n_epochs': 2000, 'milestones': [500], 'update_int': 100, 'init_lr': .01,
                                  'cp_epochs': list(range(0, 2000, 100)) + [1999]} for _ in range(1)]

# ======================================================================================================================
# Save the parameters
# ======================================================================================================================

save_path = pathlib.Path(ps['param_save_dir']) / ps['param_filename']
with open(save_path, 'wb') as f:
    pickle.dump(ps, f)

print('Parameters saved to: ' + str(save_path))
