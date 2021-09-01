""" Contains tools for synthesizing Gaussian non-linear regression models with Ahrens lab data.

Specifically, the tools in this module are intended to work with the data from the Chen et al., Neuron 2018 paper.
"""

import glob
from typing import List, Optional, Sequence, Tuple, Union
import pathlib
import pickle

import numpy as np
import torch

from ahrens_wbo.data_processing import generate_torch_dataset
from ahrens_wbo.data_processing import load_and_preprocess_data
from ahrens_wbo.data_processing import SegmentTable

from janelia_core.stats.regression import r_squared

from probabilistic_model_synthesis.gaussian_nonlinear_regression import fit_with_hypercube_priors
from probabilistic_model_synthesis.gaussian_nonlinear_regression import predict
from probabilistic_model_synthesis.gaussian_nonlinear_regression import VICollection
from probabilistic_model_synthesis.utilities import print_heading
from probabilistic_model_synthesis.utilities import print_info


def find_period_time_points(cand_ts, period, shock: bool, labels: dict) -> np.ndarray:
    """ Finds data points of a given period.

    Args:

        cand_ts: An array of time stamps that we search through to see if any are in the requested period.

        period: The period we want to search for

        shock: True of False depending on if the period should have shock applied

        labels: The labelled time points of the original experiment, as produced by label_periods or label_subperiods

    Returns:

        inds: Indices in cand_ts of points that are in the requested period.
    """
    period_ts = [labels['ts'][sl['slice']] for sl in labels['labels'][period]
                 if sl['shock'] == shock]
    period_ts = np.concatenate(period_ts)
    _, inds, _ = np.intersect1d(cand_ts, period_ts, return_indices=True)
    return inds


def syn_ahrens_gnlr_mdls(fold_str_dir: str, fold_str_file: str, segment_table_dir: str, segment_table_file: str,
                         data_dir: str, fold: Union[int, str], mdl_opts: dict, sp_cp_dir: Optional[str] = None,
                         ip_cp_dir: Optional[str] = None, normalize_beh_vars: bool = True, neural_gain: float = 10000.0,
                         beh_gain: float = 100.0, z_ratio: float = 2.5,
                         subject_filter: Optional[Sequence[int]] = None) -> Tuple[dict, List]:
    """
    Synthesizes non-linear regression models which predict behavior from neural activity.

    This function performs basic data pre-processing and then calls fit_with_hypercube_priors to do the actual fitting.

    Args:

        fold_str_dir: The directory containing the fold structure file

        fold_str_file: The file containing the fold structures to process

        segment_table_dir: The directory containing the segment table file

        segment_table_file: The segment table file

        data_dir: Directory with the datasets in it

        fold: The fold of data in the fold structures that we are fitting to.  Depending on the fold structure this
        should be specified as an integer (for folds identified with numbers) or a string (when folds are identified
        with strings).

        mdl_opts: Dictionary of parameters specifying model structure and fitting to be passed to the function
        fit_with_hypercube_priors.  All options can be specified except data, and props (which are determined by
        the subjects fit).  See that function for list of options which must be provided.

        sp_cp_dir: Directory to save check points in for shared posterior fitting.  Can be None, if no check points will
        be created.

        ip_cp_dir: Directory to save check points in for individual posterior fitting.  Can be None, if no check points
        will be created.

        normalize_beh_vars: True if behavioral variables should be normalized.

        neural_gain: The gain to apply to neural signals.  Applying a gain can be help avoid issues with floating
        point underflow.

        beh_gain: The gain to apply to behavioral signals.

        z_ratio: Ratio of z-plane spacing to voxel side length in x-y dimensions

        subject_filter: A list of subject numbers to fit to.  This can be any subset of the subjects in the fold
        structures file. If None, all subjects in the fold structures file will be fit.

        ps: A dictionary with parameters for

    Returns:

        fit_rs: The output of the call to fit_with_hypercube_priors.  See that function for more details.

        subject_order: Gives the subject numbers for the corresponding fit models in fit_rs.

    """

    # ======================================================================================================================
    # Load the fold structure file
    # ======================================================================================================================

    print_heading('Loading the fold structure.')

    fold_str_path = pathlib.Path(fold_str_dir) / fold_str_file
    print_info('Loading fold structure from file: ' + str(fold_str_path))

    with open(fold_str_path, 'rb') as f:
        fold_groups = pickle.load(f)

    # ======================================================================================================================
    # Load segment tables
    # ======================================================================================================================

    print_heading('Loading segment tables.')
    segment_table_path = pathlib.Path(segment_table_dir) / segment_table_file
    print_info('Loading segment tables from: ' + str(segment_table_path))

    with open(segment_table_path, 'rb') as f:
        segment_file_data = pickle.load(f)
        segment_tables = segment_file_data['segment_tables']
        for s_n in segment_tables.keys():
            segment_tables[s_n] = SegmentTable.from_dict(segment_tables[s_n])

    # ======================================================================================================================
    # Apply the subject filter if needed
    # ======================================================================================================================

    if subject_filter is not None:
        print_heading('Applying subject filter.')
        new_fold_groups = {s_n: fold_groups[s_n] for s_n in subject_filter}
        fold_groups = new_fold_groups

    # ======================================================================================================================
    # Load data and do basic preprocessing
    # ======================================================================================================================

    print_heading('Loading data.')
    subjects = list(fold_groups.keys())
    subject_data, subject_neuron_locs = load_and_preprocess_data(data_folder=data_dir, subjects=subjects,
                                                                 normalize_beh_vars=normalize_beh_vars,
                                                                 neural_gain=neural_gain, beh_gain=beh_gain,
                                                                 z_ratio=z_ratio)

    # ======================================================================================================================
    # Extract training data
    # ======================================================================================================================

    print_heading('Forming training data for each subject.')

    train_data = dict()
    for s_n in subjects:
        subject_train_slices = segment_tables[s_n].find_all(fold_groups[s_n][fold]['train'])
        train_data[s_n] = generate_torch_dataset(dataset=subject_data[s_n], slices=subject_train_slices,
                                                 ts_keys=['dff', 'behavior'], inc_time_stamps=True)

    # Convert training data from TimeSeriesBatch objects to Torch tensors
    train_data = {s_n: (ts_b.data[0][ts_b.i_x, :], ts_b.data[1][ts_b.i_y, :]) for s_n, ts_b in train_data.items()}

    # ======================================================================================================================
    # Perform model synthesis
    # ======================================================================================================================

    print_heading('Synthesizing models.')

    # Specify the ordering of subjects when we go from dictionaries to lists
    subject_order = list(train_data.keys())

    # Add check point directory to fitting options
    for d_i, d in enumerate(mdl_opts['sp_fit_opts']):
        d['cp_save_folder'] = sp_cp_dir
        d['cp_save_str'] = 'rnd_' + str(d_i) + '_'
    for d_i, d in enumerate(mdl_opts['ip_fit_opts']):
        d['cp_save_folder'] = ip_cp_dir
        d['cp_save_str'] = 'rnd_' + str(d_i) + '_'

    fit_rs = fit_with_hypercube_priors(data=[train_data[s_n] for s_n in subject_order],
                                       props=[subject_neuron_locs[s_n] for s_n in subject_order],
                                       **mdl_opts)

    # ======================================================================================================================
    # Return results
    # ======================================================================================================================

    return fit_rs, subject_order


def post_process(results_file: str, early_stopping_subjects: Sequence[int] = None, eval_types: Sequence[str] = None,
                 early_stopping: bool = True):
    """ Post-processes results produced by syn_ahrens_gnlr_mdls.

    This function will:

        1) Preprocess data

        2) Break data up into train, validation and test sets

        3) Measure model performance (with r-squared averaged across behavioral channels) for all saved check
        points on train, validation and test data for both sp and ip model fits.

        4) If requested, pick the best model with retrospective early stopping.  If not using early stopping,
        the best model is the last trained model.

        5) Produce predictions for the best sp and ip models.

    Note: This function assumes that data and check points have not been moved since model fitting.

    Args:

        results_file: Full path to file containing saved results.

        early_stopping_subjects: Subjects to base early stopping on

        eval_types: The type of fits to evaluate ('sp' or 'ip').  If None, defaults to ['sp', 'ip'].

        early_stopping: True if early stopping should be used.

    Returns:

        d: A dictionary with the following fields:

            ps: A dictionary recording the values of the arguments of this function

            subject_order: A numpy array given the ordering of subject numbers of relevant parts of the post-processed
            results

            fit_ps: the parameters that were present with fitting the models

            rs: A dictionary with the keys 'sp' and 'ip' (if results are only requested for one model type, only one
            of these keys will be present).  The values of these keys are themselves dictionary which hold the
            post-processed results for the respective fit type.  These dictionaries will have the following keys:

                cp_results: A dictionary with the results across checkpoints. It will have the keys:

                    epochs: The epochs that check points were created after

                    train_perf: A numpy array of size num_check_points*n_subjects containing the performance on
                    training data at each checkpoint for each model.  Models are listed in the order given by
                    subject_order.

                    validation_perf, test_perf: Numpy arrays, analagous to train_perf holding performance on
                    validation and test data.

                early_stopping: A dicionary with the field best_cp_ind: which gives the index of the check point with
                the best performance.  If early stopping was not used, this will be None.

                preds: A dictionary with keys, 'train', 'validation', and 'test' containing true and predicted values
                for each type of data.  Each value for each of these keys will itself be a dictionary, with the keys
                'y', 'y_hat', 't' containing the true behavioral signals, the predicted behavioral signals and the
                time points of the signals, respectively.

    """

    # ==================================================================================================================
    # Define helper functions
    # ==================================================================================================================
    def _process_fit_type_rs(mdl_type: str, fit_rs: dict, fit_ps: dict, subject_order: Sequence[int],
                             early_stopping_subjects: Sequence[int], data: dict, subject_neuron_locs: dict,
                             early_stopping: bool):
        """
        Args:

            mdl_type: The fit type to process: either 'sp' or 'ip'

            fit_rs: The fit results for the model type.

            fit_ps: The saved parameters providing during fitting

            subject_order: The order of subjects in the saved fitting results

            early_stopping_subjects: The subjects to use for early stopping

            data: The train, validation and test data for each subject, as loaded in the main function

            subject_neuron_locs: Locations of neurons for each subject, as loaded in the main function

            early_stopping: True if early stopping should be used
        """

        # ==============================================================================================================
        # Load check points

        cp_dir = pathlib.Path(fit_ps[mdl_type + '_cp_dir'])
        cp_files = glob.glob(str(cp_dir / 'cp_*.pt'))
        n_cps = len(cp_files)
        cp_rs = [None] * n_cps
        for cp_i, cp_file in enumerate(cp_files):
            cp_rs[cp_i] = torch.load(cp_file)
            print('Done loading check point ' + str(cp_i + 1) + ' of ' + str(n_cps) + '.')

        # ==============================================================================================================
        # Sort check points by epoch
        cp_epochs = np.asarray([cp['total_epoch'] for cp in cp_rs])
        cp_sort_order = np.argsort(cp_epochs)
        cp_epochs = cp_epochs[cp_sort_order]
        cp_rs = [cp_rs[i] for i in cp_sort_order]

        # ==============================================================================================================
        # Evaluate performance on train, validation and test data for each subject at each check point
        n_subjects = len(subject_order)
        n_cps = len(cp_epochs)

        train_cp_perf = np.zeros([n_cps, n_subjects])
        validation_cp_perf = np.zeros([n_cps, n_subjects])
        test_cp_perf = np.zeros([n_cps, n_subjects])

        perf_arrays = [train_cp_perf, validation_cp_perf, test_cp_perf]
        cv_strings = ['train', 'validation', 'test']

        for cp_i, cp in enumerate(cp_rs):
            for s_i, s_n in enumerate(subject_order):
                for perf_array, cv_string in zip(perf_arrays, cv_strings):
                    if data[s_n][cv_string] is not None:
                        x = data[s_n][cv_string][0]
                        y = data[s_n][cv_string][1].numpy()
                        coll = VICollection.from_checkpoint(cp['vi_collections'][s_i], props=subject_neuron_locs[s_n])
                        y_hat = predict(coll, x).detach().numpy()
                        perf_array[cp_i, s_i] = np.mean(r_squared(truth=y, pred=y_hat))

        # ==============================================================================================================
        # Perform early stopping if we are suppose to
        if early_stopping:
            validation_subject_inds = np.asarray([np.argwhere(np.asarray(subject_order) == s_n)[0][0]
                                                  for s_n in early_stopping_subjects])

            mean_validation_performance = np.mean(validation_cp_perf[:, validation_subject_inds], axis=1)

            best_cp_ind = np.argmax(mean_validation_performance)
            best_cp = cp_rs[best_cp_ind]

            best_vi_collections = [VICollection.from_checkpoint(best_cp['vi_collections'][s_i],
                                                                props=subject_neuron_locs[s_n])
                                   for s_i, s_n in enumerate(subject_order)]

        else:
            best_cp_ind = None
            best_vi_collections = fit_rs['vi_collections']

        # ==============================================================================================================
        # Generate predictions with the best synthesized models
        preds = dict()
        for s_i, s_n in enumerate(subject_order):
            subj_preds = dict()
            for cv_string in cv_strings:
                if data[s_n][cv_string] is not None:
                    x = data[s_n][cv_string][0]
                    y = data[s_n][cv_string][1].numpy()
                    t = data[s_n][cv_string][2].numpy()
                    y_hat = predict(best_vi_collections[s_i], x).detach().numpy()

                    subj_preds[cv_string] = {'y': y, 'y_hat': y_hat, 't': t}
                else:
                    subj_preds[cv_string] = None
            preds[s_n] = subj_preds

        # ==============================================================================================================
        # Package everything together and return

        return {'cp_results': {'epochs': cp_epochs,
                               'train_perf': train_cp_perf,
                               'validation_perf': validation_cp_perf,
                               'test_perf': test_cp_perf},
                'early_stopping': {'best_cp_ind': best_cp_ind},
                'preds': preds}

    # ==================================================================================================================
    # Load the results
    # ==================================================================================================================
    rs = torch.load(results_file)
    fit_ps = rs['ps']
    subject_order = rs['subject_order']
    fold = fit_ps['fold']

    # ==================================================================================================================
    # Provide default values
    # ==================================================================================================================

    if eval_types is None:
        eval_types = ['ip', 'sp']

    if early_stopping_subjects is None:
        early_stopping_subjects = subject_order

    # ==================================================================================================================
    # Load segment tables, fold structures and data
    # ==================================================================================================================

    # Load segment tables
    segment_table_path = pathlib.Path(fit_ps['segment_table_dir']) / fit_ps['segment_table_file']
    with open(segment_table_path, 'rb') as f:
        st_data = pickle.load(f)
        segment_tables = st_data['segment_tables']
    segment_tables = {k: SegmentTable.from_dict(v) for k, v in segment_tables.items()}

    # Load fold structures
    fold_str_path = pathlib.Path(fit_ps['fold_str_dir']) / fit_ps['fold_str_file']
    with open(fold_str_path, 'rb') as f:
        fold_groups = pickle.load(f)

    # Load data
    subject_data, subject_neuron_locs = load_and_preprocess_data(data_folder=fit_ps['data_dir'],
                                                                 subjects=subject_order,
                                                                 normalize_beh_vars=fit_ps['normalize_beh_vars'],
                                                                 neural_gain=fit_ps['neural_gain'],
                                                                 beh_gain=fit_ps['beh_gain'],
                                                                 z_ratio=fit_ps['z_ratio'])

    # ==================================================================================================================
    # Form validation, train and test data
    # ==================================================================================================================
    data = dict()
    for s_n in subject_order:
        subject_train_slices = segment_tables[s_n].find_all(fold_groups[s_n][fold]['train'])
        train_data_n = generate_torch_dataset(dataset=subject_data[s_n],
                                              slices=subject_train_slices,
                                              ts_keys=['dff', 'behavior'], inc_time_stamps=True)

        train_data_n = (train_data_n.data[0][train_data_n.i_x, :],
                        train_data_n.data[1][train_data_n.i_y, :],
                        train_data_n.data[2][train_data_n.i_y])

        if fold_groups[s_n][fold]['test'] is not None:
            subject_test_slices = segment_tables[s_n].find_all(fold_groups[s_n][fold]['test'])
            test_data_n = generate_torch_dataset(dataset=subject_data[s_n],
                                                 slices=subject_test_slices,
                                                 ts_keys=['dff', 'behavior'], inc_time_stamps=True)

            test_data_n = (test_data_n.data[0][test_data_n.i_x, :],
                           test_data_n.data[1][test_data_n.i_y, :],
                           test_data_n.data[2][test_data_n.i_y])
        else:
            test_data_n = None

        if fold_groups[s_n][fold]['validation'] is not None:
            subject_validation_slices = segment_tables[s_n].find_all(fold_groups[s_n][fold]['validation'])
            validation_data_n = generate_torch_dataset(dataset=subject_data[s_n],
                                                   slices=subject_validation_slices,
                                                   ts_keys=['dff', 'behavior'],
                                                   inc_time_stamps=True)

            validation_data_n = (validation_data_n.data[0][validation_data_n.i_x, :],
                                 validation_data_n.data[1][validation_data_n.i_y, :],
                                 validation_data_n.data[2][validation_data_n.i_y])
        else:
            validation_data_n = None

        data[s_n] = {'train': train_data_n,
                     'test': test_data_n,
                     'validation': validation_data_n}

    # ==================================================================================================================
    # Post-process sp and ip models (or just one fit type if requested)
    # ==================================================================================================================
    rs_dict = dict()
    for e_type in eval_types:
        rs_dict[e_type] = _process_fit_type_rs(mdl_type=e_type, fit_ps=fit_ps, fit_rs=rs['rs'][e_type],
                                               subject_order=subject_order,
                                               early_stopping_subjects=early_stopping_subjects, data=data,
                                               subject_neuron_locs=subject_neuron_locs, early_stopping=early_stopping)

    # ==================================================================================================================
    # Finish packaging results and return
    # ==================================================================================================================

    ps = {'results_file': results_file, 'early_stopping_subjects': early_stopping_subjects, 'eval_types': eval_types,
          'early_stopping': early_stopping}

    rs_dict['subject_order'] = subject_order
    rs_dict['fit_ps'] = fit_ps
    rs_dict['ps'] = ps

    return rs_dict

