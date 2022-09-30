""" Runs a full cross-validated, same-behavior transfer analysis.

This script will  multiple jobs to the Janelia cluster to run in parallel.  Each job will correspond to one transfer
subject with a fixed quantity of training data for the transfer and base fish.  This script loops over transfer
subjects, amounts of training data, folds of data (there are multiple folds for each amount of training data)
and also fits the transfer subjects with the base subjects as well as individually.

In addition to the initial fitting, the user can also chose to do the post processing for each fit.

"""

import os
import pathlib
import time

# ======================================================================================================================
# Parameters go here
# ======================================================================================================================

# Specify if we should run the fits
FIT = True

# Specify if we should run post processing
POST_PROCESS = True

# Name of files that fitting results should be in - all results will be saved in files of the same name, with the
# folder structure being used to denote results with different settings.  The name of files containing results from
# post processing will be of the form pp_<SAVE_FILE>.
SAVE_FILE = 'fit_results.pt'

# The base subjects, we can fit each target subject with, for the transfer analyses
BASE_SUBJECTS = [1, 2, 5, 6]

# Each of the target subjects
TGT_SUBJECTS = [8, 10, 11]

# Specify the full path to the parameter file
PARAM_FILE = r'/groups/fitzgerald/fitzgeraldlab/bishoplab/projects/probabilistic_model_synthsis/results/publication_results/gnlr/real_data/fit_params.pkl'

# Specify the base folder into which results should be saved
RESULTS_DIR = r'/groups/fitzgerald/fitzgeraldlab/bishoplab/projects/probabilistic_model_synthsis/results/publication_results/gnlr/real_data'

# Specify the fold structure files we should fit to
FOLD_STR_FILES = ['fold_str_base_14_tgt_1.pkl',
                  'fold_str_base_14_tgt_2.pkl',
                  'fold_str_base_14_tgt_4.pkl',
                  'fold_str_base_14_tgt_8.pkl',
                  'fold_str_base_14_tgt_14.pkl']

# Specify how many folds we fit
N_FOLDS = 3

# Specify number of slots for each analysis type
COMB_N_SLOTS = 3
IND_N_SLOTS = 3

# ======================================================================================================================
# Code to submit the jobs goes here
# ======================================================================================================================
TYPES = ['comb', 'ind']
GPU_QUEUE = 'gpu_rtx'

BASE_COMB_CALL = 'bsub -n ' + str(COMB_N_SLOTS) + ' -gpu "num=1"' + ' -q ' + GPU_QUEUE
BASE_IND_CALL = 'bsub -n ' + str(IND_N_SLOTS) + ' -gpu "num=1"' + ' -q ' + GPU_QUEUE
print(BASE_COMB_CALL)
ANACONDA_SETUP = '. /groups/bishop/home/bishopw/anaconda3/etc/profile.d/conda.sh'
ENV_SETUP = 'conda activate unified_env'

BASE_FIT_COMMAND = 'python /groups/fitzgerald/fitzgeraldlab/bishoplab/projects/probabilistic_model_synthsis/code/probabilistic_model_synthesis/real_data/gnlr/syn_ahrens_gnlr_mdls.py'
BASE_PP_COMMAND = 'python /groups/fitzgerald/fitzgeraldlab/bishoplab/projects/probabilistic_model_synthsis/code/probabilistic_model_synthesis/real_data/gnlr/post_process.py'

rand_seed = 0
for fold_str_file in FOLD_STR_FILES:
    fold_str_dir = pathlib.Path(RESULTS_DIR) / pathlib.Path(fold_str_file).stem
    for fold in range(N_FOLDS):
        fold_dir = fold_str_dir / ('fold_' + str(fold))
        for tgt_subj in TGT_SUBJECTS:
            tgt_subj_dir = fold_dir / ('subj_' + str(tgt_subj))
            for fit_type in TYPES:
                type_dir = tgt_subj_dir / fit_type
                if not os.path.isdir(type_dir):
                    os.makedirs(type_dir)

                # Build up the fitting call here
                sp_cp_dir = type_dir / 'sp_cp'
                ip_cp_dir = type_dir / 'ip_cp'

                base_print_str = ('Submitting job for fold structure file ' + fold_str_file + ', fold ' + str(fold) +
                                  ', target subject ' + str(tgt_subj))
                if fit_type == 'comb':
                    call = BASE_COMB_CALL
                    fit_subjs = ','.join([str(s) for s in BASE_SUBJECTS]) + ',' + str(tgt_subj)
                    print(base_print_str + ', combined')
                else:
                    call = BASE_IND_CALL
                    fit_subjs = str(tgt_subj)
                    print(base_print_str + ', individual')

                rand_seed += 1
                if FIT:
                    job_command = BASE_FIT_COMMAND + ' ' + PARAM_FILE
                    job_command += ' -results_dir ' + str(type_dir)
                    job_command += ' -fold_str_file ' + fold_str_file
                    job_command += ' -fold ' + str(fold)
                    job_command += ' -sp_cp_dir ' + str(sp_cp_dir)
                    job_command += ' -ip_cp_dir ' + str(ip_cp_dir)
                    job_command += ' -subject_filter ' + fit_subjs
                    job_command += ' -save_file ' + SAVE_FILE
                    job_command += ' -rand_seed ' + str(rand_seed)
                else:
                    job_command = ''

                if POST_PROCESS:
                    if FIT:
                        job_command += ' && '
                    results_file_path = str(type_dir / SAVE_FILE)
                    job_command += BASE_PP_COMMAND + ' ' + results_file_path
                    save_file_path = str(type_dir / ('pp_' + pathlib.Path(SAVE_FILE).stem + '.pkl'))
                    job_command += ' ' + save_file_path
                    job_command += ' -early_stopping_subjects ' + fit_subjs
                    print('Performing early stopping using validation data from all subjects.')
                    job_command += ' -early_stopping True'
                    job_command += ' -rand_seed ' + str(rand_seed)

                call += ' -o ' + str(type_dir / 'log.txt')
                call += ' "' + ANACONDA_SETUP + ' && ' + ENV_SETUP + ' && ' + job_command + '"'

                # Run the job here
                os.system(call)
                # Wait a little while to allow for data loading
                time.sleep(60)







