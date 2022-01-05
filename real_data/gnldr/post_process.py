import argparse
import os

import torch

from janelia_core.ml.utils import list_torch_devices
from probabilistic_model_synthesis.gnldr_ahrens_tools import post_process
from probabilistic_model_synthesis.utilities import parse_bool_str
from probabilistic_model_synthesis.utilities import print_heading
from probabilistic_model_synthesis.utilities import print_info

# ======================================================================================================================
# Read in parameters
# ======================================================================================================================

parser = argparse.ArgumentParser(description=('Post processes models after fitting them with the script ' +
                                              'syn_ahrens_gnldr_mdls.py.  This is a light-weight wrapper' +
                                              'around the function fa_post_process, meant for use with' +
                                              'the Janelia cluster.'))

parser.add_argument('results_file', type=str, help='path to file holding results.')
parser.add_argument('save_file', type=str, help='path to file which should be created to save post processed results.')

parser.add_argument('-early_stopping_subjects', type=str, default=None, help=('Comma separated list of subject to ' +
                                                                              'base early stopping on.  If not ' +
                                                                              'provided, early stopping will be ' +
                                                                              'based on all fit subjects.'))

parser.add_argument('-test_periods', type=str, help=('Comma seperate list of different periods ' +
                                                                   'we want to evaluate performance on in the test ' +
                                                                   'data.'))

parser.add_argument('-ip', type=str, default='True', help='The string True if we should process ip fit results.')
parser.add_argument('-sp', type=str, default='True', help='The string True if we should process ip fit results.')
parser.add_argument('-early_stopping', type=str, default='True', help=('The string True if we should use ' +
                                                                       'early stopping.'))

args = parser.parse_args()

args = parser.parse_args()

results_file = args.results_file
save_file = args.save_file

if args.test_periods is not None:
    test_periods = args.test_periods.split(',')
else:
    test_periods = None

if args.early_stopping_subjects is not None:
    early_stopping_subjects = [int(vl) for vl in args.early_stopping_subjects.split(',')]
else:
    early_stopping_subjects = None

eval_types = []
if parse_bool_str(args.ip):
    eval_types.append('ip')
if parse_bool_str(args.sp):
    eval_types.append('sp')

early_stopping = parse_bool_str(args.early_stopping)

# Only post process results if we can't find any existing post processed results
if True: #not os.path.exists(save_file):

    # ==================================================================================================================
    # Post process results
    # ==================================================================================================================

    print_heading('Post processing results.')

    devices, _ = list_torch_devices()

    pp_rs = post_process(results_file=results_file, early_stopping_subjects=early_stopping_subjects,
                         test_periods=test_periods, eval_types=eval_types, early_stopping=early_stopping,
                         device=devices[0])

    # ==================================================================================================================
    # Save results
    # ==================================================================================================================

    print_heading('Saving results.')
    torch.save(pp_rs, save_file)
    print_info('Results save to ' + str(save_file) + '.')

else:
    print_heading('Existing post processed results found.  Skipping post processing. ' +
                  'Existng post processed file: ' + save_file)


