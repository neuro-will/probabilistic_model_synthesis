import argparse
import pickle
import random

import numpy as np
import torch


from probabilistic_model_synthesis.gnlr_ahrens_tools import post_process
from probabilistic_model_synthesis.utilities import parse_bool_str
from probabilistic_model_synthesis.utilities import print_heading
from probabilistic_model_synthesis.utilities import print_info


# ======================================================================================================================
# Read in parameters
# ======================================================================================================================

parser = argparse.ArgumentParser(description=('Post-processes results after models are synthesized with ' +
                                              'syn_ahrens_gnlr_mdls. This is a light wrapper around the function ' +
                                              'gnlr_ahrens_tools.post_process, that enables post processing to be called directly ' +
                                              'from the command line when processing things in parallel on the ' +
                                              'Janelia cluster.'))

parser.add_argument('results_file', type=str, help='path to file holding results.')
parser.add_argument('save_file', type=str, help='path to file which should be created to save post processed results.')
parser.add_argument('-early_stopping_subjects', type=str, default=None, help=('Comma separated list of subject to ' +
                                                                              'base early stopping on.  If not ' +
                                                                              'provided, early stopping will be ' +
                                                                              'based on all fit subjects.'))

parser.add_argument('-ip', type=str, default='True', help='The string True if we should process ip fit results.')
parser.add_argument('-sp', type=str, default='True', help='The string True if we should process ip fit results.')
parser.add_argument('-early_stopping', type=str, default='True', help=('The string True if we should use ' +
                                                                       'early stopping.'))

parser.add_argument('-rand_seed', type=str, default=None, help='Random seed for reproducability.  If not used, seed ' +
                                                                 'will not be set.')

args = parser.parse_args()

results_file = args.results_file
save_file = args.save_file

if args.early_stopping_subjects is not None:
    early_stopping_subjects = [int(vl) for vl in args.early_stopping_subjects.split(',')]
else:
    early_stopping_subjects = None

eval_types = []
if parse_bool_str(args.sp):
    eval_types.append('sp')
if parse_bool_str(args.ip):
    eval_types.append('ip')

if args.rand_seed is not None:
    random_seed = int(args.rand_seed)
else:
    random_seed = None

early_stopping = parse_bool_str(args.early_stopping)

# ======================================================================================================================
# Set random seed if we are suppose to
# ======================================================================================================================
if random_seed is not None:
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)

# ======================================================================================================================
# Post process results
# ======================================================================================================================

pp_rs = post_process(results_file=results_file, early_stopping_subjects=early_stopping_subjects, eval_types=eval_types,
                     early_stopping=early_stopping)

# ======================================================================================================================
# Save results
# ======================================================================================================================

print_heading('Saving results.')
with open(save_file, 'wb') as f:
    pickle.dump(pp_rs, f)
print_info('Results save to ' + str(save_file) + '.')

