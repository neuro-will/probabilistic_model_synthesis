""" Runs a full across-behavior transfer analysis.

This script will  multiple jobs to the Janelia cluster to run in parallel.  Each job will correspond to

In addition to the initial fitting, the user can also chose to do the post processing for each fit.

"""

import os
import pathlib
import time

# ======================================================================================================================
# Parameters go here
# ======================================================================================================================

# Specify if we should run the fits
FIT = False

# Specify if we should run post processing
POST_PROCESS = True

# Name of files that fitting results should be in - all results will be saved in files of the same name, with the
# folder structure being used to denote results with different settings.  The name of files containing results from
# post processing will be of the form pp_<SAVE_FILE>.
SAVE_FILE = 'test_results.pt'

# Each of the target subjects
TGT_SUBJECTS = [8, 9, 11]

# Specify the full path to the parameter file
PARAM_FILE = r'/groups/bishop/bishoplab/projects/probabilistic_model_synthesis/results/real_data/gnldr/across_cond_transfer_analysis/v0/transfer_params.pkl'

# Specify the base folder into which results should be saved
RESULTS_DIR = r'/groups/bishop/bishoplab/projects/probabilistic_model_synthesis/results/real_data/gnldr/across_cond_transfer_analysis/v0'

# String prepended to all files containing the fold structure we will be working with
FOLD_STR_PRE_STR = 'ac_an'

# String appended to all files containing the fold structures we will be working with
FOLD_STR_APP_STR = 'folds.pkl'

# Specify which folds we fit to - folds are broken up by the training condition for the target fish
FOLDS = ['omr_l_ns', 'omr_r_ns', 'omr_f_ns']

# Specify the periods we want to measure performance on individually in the test data
TEST_PERIODS = ['omr_forward', 'omr_left', 'omr_right']

# Specify job resources for fitting models
N_SLOTS = 3
QUEUE = 'gpu_rtx'
N_GPU = 1

# ======================================================================================================================
# Code to submit the jobs goes here
# ======================================================================================================================
TYPES = ['multi_cond', 'single_cond']

BASE_CALL = 'bsub -n ' + str(N_SLOTS) + ' -gpu "num=' + str(N_GPU) + '"' + ' -q ' + QUEUE

ANACONDA_SETUP = '. /groups/bishop/home/bishopw/anaconda3/etc/profile.d/conda.sh'
ENV_SETUP = 'conda activate probabilistic_model_synthesis'

BASE_FIT_COMMAND = 'python /groups/bishop/bishoplab/projects/probabilistic_model_synthesis/code/probabilistic_model_synthesis/real_data/gnldr/syn_ahrens_gnldr_mdls.py'
BASE_PP_COMMAND = 'python /groups/bishop/bishoplab/projects/probabilistic_model_synthesis/code/probabilistic_model_synthesis/real_data/gnldr/post_process.py'

for fold in FOLDS:
    fold_str_dir = pathlib.Path(RESULTS_DIR) / fold
    for tgt_subj in TGT_SUBJECTS:
        tgt_subj_dir = fold_str_dir / ('subj_' + str(tgt_subj))
        for fit_type in TYPES:
            type_dir = tgt_subj_dir / fit_type
            fold_str_file = FOLD_STR_PRE_STR + '_tgt_' + str(tgt_subj) + '_' + fit_type + '_' + FOLD_STR_APP_STR
            if not os.path.isdir(type_dir):
                os.makedirs(type_dir)

            # Build up the fitting call here
            sp_cp_dir = type_dir / 'sp_cp'
            ip_cp_dir = type_dir / 'ip_cp'

            print('Submitting job for fold ' + fold + ', target subject ' + str(tgt_subj) +
                  ' and type ' + fit_type + '.')

            if FIT:
                job_command = BASE_FIT_COMMAND + ' ' + PARAM_FILE
                job_command += ' -results_dir ' + str(type_dir)
                job_command += ' -fold_str_file ' + fold_str_file
                job_command += ' -fold ' + fold
                job_command += ' -sp_cp_dir ' + str(sp_cp_dir)
                job_command += ' -ip_cp_dir ' + str(ip_cp_dir)
                job_command += ' -save_file ' + SAVE_FILE
            else:
                job_command = ''

            if POST_PROCESS:
                if FIT:
                    job_command += ' && '
                results_file_path = str(type_dir / SAVE_FILE)
                job_command += BASE_PP_COMMAND + ' ' + results_file_path
                save_file_path = str(type_dir / ('pp_' + pathlib.Path(SAVE_FILE).stem + '.pt'))
                job_command += ' ' + save_file_path
                job_command += ' -early_stopping_subjects ' + str(tgt_subj)
                job_command += ' -early_stopping True'
                job_command += ' -test_periods ' + ','.join(TEST_PERIODS)

            call = BASE_CALL
            call += ' -o ' + str(type_dir / 'log.txt')
            call += ' "' + ANACONDA_SETUP + ' && ' + ENV_SETUP + ' && ' + job_command + '"'

            # Run the job here
            os.system(call)
            # Wait a little while to allow for data loading
            time.sleep(60)







