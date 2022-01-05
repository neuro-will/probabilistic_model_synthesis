""" Contains tools for synthesizing Gaussian non-linear dimensionality reduction models with Ahrens lab data.

Specifically, the tools in this module are intended to work with the data from the Chen et al., Neuron 2018 paper.
"""

import copy
import glob
from typing import List, Optional, Sequence, Tuple
import pathlib
import pickle

import numpy as np
import torch

from ahrens_wbo.annotations import label_subperiods
from ahrens_wbo.data_processing import generate_torch_dataset
from ahrens_wbo.data_processing import load_and_preprocess_data
from ahrens_wbo.data_processing import SegmentTable

from probabilistic_model_synthesis.fa import orthonormalize
from probabilistic_model_synthesis.gaussian_nonlinear_dim_reduction import approximate_elbo
from probabilistic_model_synthesis.gaussian_nonlinear_dim_reduction import evaluate_check_points
from probabilistic_model_synthesis.gaussian_nonlinear_dim_reduction import infer_latents
from probabilistic_model_synthesis.gaussian_nonlinear_dim_reduction import synthesize_fa_mdls
from probabilistic_model_synthesis.gaussian_nonlinear_dim_reduction import PriorCollection
from probabilistic_model_synthesis.gaussian_nonlinear_dim_reduction import VICollection
from probabilistic_model_synthesis.gnlr_ahrens_tools import find_period_time_points
from probabilistic_model_synthesis.utilities import print_heading
from probabilistic_model_synthesis.utilities import print_info


def post_process(results_file: str, early_stopping_subjects: Sequence[int] = None, test_periods: List[str] = None,
                 eval_types: Sequence[str] = None, early_stopping: bool = True, device: torch.device = None):
    """ Post-processes results produced by syn_ahrens_fa_mdls.

    This function will:

        1) Preprocess data

        2) Break data up into train, validation and test sets

        3) Measure model performance (with the ELBO) for all saved check
        points on train, validation and test data for both sp and ip model fits.

        4) If requested, pick the best model with retrospective early stopping.  If not using early stopping,
        the best model is the last trained model.

        5) Estimate latents for the best sp and ip models.

        6) Report the ELBO for the best model on train, validation and test data for all subjects.

    Note: This function assumes that data and check points have not been moved since model fitting.

    Args:

        results_file: Full path to file containing saved results.

        early_stopping_subjects: Subjects to base early stopping on

        test_periods: A list of periods to measure performance separately on in the test data. If None,
        performance will not be measured on separate periods in the test data. The periods provided
        here should correspond to period labels produced by the function label_subperiods()

        eval_types: The type of fits to evaluate ('sp' or 'ip').  If None, defaults to ['sp', 'ip'].

        early_stopping: True if early stopping should be used.

        device: Device to use for post-processing.  If None, cpu will be used.

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

                    files: The files containing the models for the checkpoints at each epoch in epoch

                    cp_perf: A dictionary with the keys 'train', 'validation' and 'test', each containing a numpy
                    array of size n_check_pts*n_subjects with performance of the checkpoints on the corresponding
                    type of data. The entry at (i, m) contains the performance for model subject_order[m] at epoch
                    epochs[i].

                    cp_logs: cp_logs[cp_i][mdl_j] is contains the log for inferring latents (as part of calculating the
                    ELBO) for check point at epoch epochs[i] and the model for subject_order[j].

                early_stopping: A dictionary with the field best_cp_ind: which gives the index of the check point with
                the best performance.  If early stopping was not used, this will be None.

                latents: A dictionary with the keys corresponding to subjects.  Each entry is itself a dictionary with
                the keys 'train', 'validation' and 'test', giving estimated latents for each data point in the
                corresponding type of data. Latents are estimated for the 'evaluation check point,' which will be picked
                using early stopping if the user specifies that or is simply the last fit model. The values associated
                with each key of thes dictionaries are themselves also dictionaries with the fields:

                    'ts': The time stamps in the original data for each data point

                    'posts': The posterior distributions over the latents.

                    'log': The fitting log for inferring latents

                    'mns': The means of the posteriors for the latents as numpy arrays.

                    'mns_o': The orthonormalized means as numpy arrays.

                lm_o: A dictionary with keys corresponding to subects.  Each key holds the orthonormalized version of
                the posterior mean for the loading matrix for the subject. This will be the loading matrices for the
                evaluation check point.

                elbo_vls: A dictionary with keys corresponding to subjects.  Each entry is itself a dictionary with the
                keys 'train', 'validation' and 'test' holding ELBO values for the evaluation check point for the
                corresponding type of data.  The values of these will be the output of the function approximate_elbo.
                See that function for more details.

                period_elbo_vls: A dictionary with keys corresponding to subjects.  If performance was not measured
                on separate periods in the test data this will be None.  If performance was measured, each entry is
                a dictionary keys corresponding to periods.  The values of the keys will be the performance
                measured on the test data of that period.  If no data of that period existed, the valuew will be None.
                If there was no test data for a subject then the entry in period_elbo_vls for that subject will be
                None.

    """

    # ==================================================================================================================
    # Define helper functions
    # ==================================================================================================================
    def _process_fit_type_rs(mdl_type: str, fit_rs: dict, fit_ps: dict, subject_order: Sequence[int], labels: dict,
                             test_periods: List, early_stopping_subjects: Sequence[int], data: dict,
                             subject_neuron_locs: dict, early_stopping: bool, device: torch.device = None,
                             eval_cp_opts: dict = None, latent_fit_opts: dict = None, n_elbo_smps: int = 100):

        CV_STRS = ['train', 'validation', 'test']

        if device is None:
            device = torch.device('cpu')

        if eval_cp_opts is None:
            eval_cp_opts = {'fit_opts': {'n_epochs': 300, 'init_lr': .1, 'milestones': [50], 'update_int': None},
                            'elbo_opts': None,
                            'n_smps': 10}

        if latent_fit_opts is None:
            latent_fit_opts = {'n_epochs': 300, 'init_lr': .1, 'update_int': None, 'milestones': [50], 'n_batches': 2}

        n_subjects = len(subject_order)
        n_latent_vars = fit_ps['mdl_opts']['n_latent_vars']

        # ==============================================================================================================
        # Examine performance across check points
        print_heading('Evaluating check point performance.')
        cp_perf = dict()
        cp_logs = dict()
        for cv_string in CV_STRS:
            print_info('Evaluating check points on ' + cv_string + ' data.')

            # Note, cp_epochs and cp_files is the same for iterations of the loop, so it's ok we reference them later
            # in the code

            cp_data = [data[s_n][cv_string][0] if data[s_n][cv_string] is not None else None
                       for s_n in subject_order]

            cp_epochs, cp_perf[cv_string], cp_files, cp_logs[cv_string] = evaluate_check_points(
                cp_folder=pathlib.Path(fit_ps[mdl_type + '_cp_dir']),
                data=cp_data,
                props=[subject_neuron_locs[s_n] for s_n in subject_order],
                device=device, **eval_cp_opts)

        # Package check point results for saving later
        cp_results = {'epochs': cp_epochs, 'files': cp_files, 'cp_perf': cp_perf, 'cp_logs': cp_logs}

        # ==============================================================================================================
        # Pick best check point based on performance, if requested
        if early_stopping:
            print_heading('Picking best check point for evaluating performance based on validation data.')

            validation_subject_inds = np.asarray([np.argwhere(np.asarray(subject_order) == s_n)[0][0]
                                                  for s_n in early_stopping_subjects])

            cp_avg = np.nanmean(cp_perf['validation'][:, validation_subject_inds], axis=1)
            best_cp_ind = np.argmax(cp_avg)

            print_info('Best CP Epoch: ' + str(cp_epochs[best_cp_ind]))
            best_cp_file = cp_files[best_cp_ind]
            best_cp = torch.load(best_cp_file)
            eval_vi_colls = [VICollection.from_checkpoint(coll) for coll in best_cp['vi_collections']]
            eval_priors = PriorCollection.from_checkpoint(best_cp['priors'])

            # Package check point results for saving later
            early_stopping = {'best_cp_ind': best_cp_ind}

        else:
            print_heading('Not using early stopping.  Evaluating performance of for the final fit models.')
            eval_vi_colls = fit_rs['vi_collections']
            eval_priors = fit_rs['priors']

            early_stopping = None

        # Repopulate properties of the selected vi collections for evaluation
        for s_i, s_n in enumerate(subject_order):
            eval_vi_colls[s_i].props = subject_neuron_locs[s_n]

        # ==============================================================================================================
        # Infer latents
        print_heading('Infering latents for the train, validation and test data with evaluation check point.')

        latents = dict()
        for s_i, s_n in enumerate(subject_order):
            sub_latents = dict()
            for cv_string in CV_STRS:
                if data[s_n][cv_string] is not None:
                    latents_post, latent_log = infer_latents(n_latent_vars=n_latent_vars, vi_collection=eval_vi_colls[s_i],
                                                             data=data[s_n][cv_string][0], fit_opts=latent_fit_opts,
                                                             device=device, distribute_data=True)

                    sub_latents[cv_string] = {'ts': data[s_n][cv_string][1].numpy(),
                                              'posts': latents_post,
                                              'log': latent_log}
                    print_info('Done inferring latents for subject ' + str(s_n) + ', ' + cv_string + '.')
                else:
                    sub_latents[cv_string] = None
            latents[s_n] = sub_latents

        # ==============================================================================================================
        # Orthonormalize latents

        print_heading('Orthonormalizing latents.')
        lm_conc = np.concatenate([eval_vi_colls[s_i].posteriors.lm_post(subject_neuron_locs[s_n]).detach().numpy()
                                  for s_i, s_n in enumerate(subject_order)], axis=0)
        for s_n in subject_order:
            for cv_string in CV_STRS:
                if latents[s_n][cv_string] is not None:
                    mns = latents[s_n][cv_string]['posts'].mns.detach().numpy()
                    _, mns_o = orthonormalize(lm=lm_conc, latents=mns)
                    latents[s_n][cv_string]['mns'] = mns
                    latents[s_n][cv_string]['mns_o'] = mns_o

        # Get the orthonormalized version of the loading matrices
        lm_conc_o, _ = orthonormalize(lm=lm_conc)
        lm_o = dict()
        cur_start = 0
        for s_n in subject_order:
            n_vars_n = data[s_n]['train'][0].shape[1]
            lm_o[s_n] = lm_conc_o[cur_start:cur_start + n_vars_n]
            cur_start += n_vars_n

        # ==============================================================================================================
        # Calculate ELBO on train, validation and test data for evaluation checkpoint
        print_heading('Calculating ELBO for train, validation and test data for evaluation check point.')

        elbo_vls = dict()
        for s_i, s_n in enumerate(subject_order):
            eval_coll_i = copy.deepcopy(eval_vi_colls[s_i])
            subj_elbo = dict()
            for cv_string in CV_STRS:
                if data[s_n][cv_string] is not None:
                    eval_coll_i.data = data[s_n][cv_string][0]
                    eval_coll_i.posteriors.latent_post = latents[s_n][cv_string]['posts']
                    eval_coll_i.to(device)
                    eval_priors.to(device)
                    with torch.no_grad():
                        cur_elbo = approximate_elbo(coll=eval_coll_i, priors=eval_priors, n_smps=n_elbo_smps)
                        subj_elbo[cv_string] = {'vl': cur_elbo, 'n_smps': eval_coll_i.data.shape[0]}
                    print('Done estimating ELBO for subject ' + str(s_n) + ', ' + cv_string + '.')
                else:
                    subj_elbo[cv_string] = None
            eval_coll_i.to(torch.device('cpu'))
            elbo_vls[s_n] = subj_elbo
        eval_priors.to(torch.device('cpu'))

        # ==============================================================================================================
        # Calculate ELBO in different periods of test data
        if test_periods is not None:
            period_elbo_vls = dict()
            for s_i, s_n in enumerate(subject_order):
                period_elbo_vls[s_n] = dict()
                if data[s_n]['test'] is not None:
                    test_ts = data[s_n]['test'][1].numpy()
                    for test_period in test_periods:
                        test_inds = find_period_time_points(cand_ts=test_ts,
                                                        period=test_period,
                                                        shock=False,
                                                        labels=labels[s_n])
                        if len(test_inds) > 0:
                            eval_coll_i = copy.deepcopy(eval_vi_colls[s_i])
                            eval_coll_i.data = data[s_n]['test'][0][test_inds, :]
                            latent_post = copy.deepcopy(latents[s_n]['test']['posts'])
                            latent_post.n = len(test_inds)
                            latent_post.mns.data = latent_post.mns.data[test_inds, :]
                            eval_coll_i.posteriors.latent_post = latent_post
                            eval_coll_i.to(device)
                            eval_priors.to(device)
                            with torch.no_grad():
                                cur_elbo = approximate_elbo(coll=eval_coll_i, priors=eval_priors,
                                                            n_smps=n_elbo_smps)
                                period_elbo_vls[s_n][test_period] = {'vl': cur_elbo,
                                                                     'n_smps': eval_coll_i.data.shape[0]}
                            eval_coll_i.to(torch.device('cpu'))
                        else:
                            period_elbo_vls[s_n][test_period] = None
                else:
                    period_elbo_vls[s_n] = None
        else:
            period_elbo_vls = None

        # ==============================================================================================================
        # Package results
        return {'cp_results': cp_results, 'early_stopping': early_stopping, 'latents': latents, 'lm_o': lm_o,
                'elbo_vls': elbo_vls, 'period_elbo_vls': period_elbo_vls, 'logs': fit_rs['logs']}

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

    if device is None:
        device = torch.device('cpu')
    print('Using device: ' + str(device))

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
                                                                 neural_gain=fit_ps['neural_gain'],
                                                                 z_ratio=fit_ps['z_ratio'])

    # ==================================================================================================================
    # Label the periods in each dataset
    # ==================================================================================================================
    labels = {s_n: {'ts': subject_data[s_n].ts_data['stim']['ts'],
                    'labels': label_subperiods(subject_data[s_n].ts_data['stim']['vls'][:])} for s_n in subject_order}

    # ==================================================================================================================
    # Form train, validation and test data
    # ==================================================================================================================

    data = dict()
    for s_n in subject_order:
        subject_train_slices = segment_tables[s_n].find_all(fold_groups[s_n][fold]['train'])
        train_data_n = generate_torch_dataset(dataset=subject_data[s_n],
                                              slices=subject_train_slices,
                                              ts_keys=['dff'], inc_time_stamps=True)

        train_data_n = (train_data_n.data[0], train_data_n.data[1])

        if fold_groups[s_n][fold]['test'] is not None:
            subject_test_slices = segment_tables[s_n].find_all(fold_groups[s_n][fold]['test'])
            test_data_n = generate_torch_dataset(dataset=subject_data[s_n],
                                                 slices=subject_test_slices,
                                                 ts_keys=['dff'], inc_time_stamps=True)

            test_data_n = (test_data_n.data[0], test_data_n.data[1])
        else:
            test_data_n = None

        if fold_groups[s_n][fold]['validation'] is not None:
            subject_validation_slices = segment_tables[s_n].find_all(fold_groups[s_n][fold]['validation'])
            validation_data_n = generate_torch_dataset(dataset=subject_data[s_n],
                                                       slices=subject_validation_slices,
                                                       ts_keys=['dff'],
                                                       inc_time_stamps=True)

            validation_data_n = (validation_data_n.data[0], validation_data_n.data[1])
        else:
            validation_data_n = None

        data[s_n] = {'train': train_data_n,
                     'test': test_data_n,
                     'validation': validation_data_n}

    # ==================================================================================================================
    # Post-process sp and ip models (or just one fit type of requested)
    # ==================================================================================================================
    rs_dict = dict()
    for e_type in eval_types:
        rs_dict[e_type] = _process_fit_type_rs(mdl_type=e_type, fit_ps=fit_ps, fit_rs=rs['rs'][e_type],
                                               subject_order=subject_order, labels=labels, test_periods=test_periods,
                                               early_stopping_subjects=early_stopping_subjects, data=data,
                                               subject_neuron_locs=subject_neuron_locs, early_stopping=early_stopping,
                                               device=device)
    # ==================================================================================================================
    # Finish packaging results and return
    # ==================================================================================================================
    ps = {'results_file': results_file, 'early_stopping_subjects': early_stopping_subjects, 'eval_types': eval_types,
          'early_stopping': early_stopping}

    rs_dict['subject_order'] = subject_order
    rs_dict['fit_ps'] = fit_ps
    rs_dict['ps'] = ps

    return rs_dict


def syn_ahrens_fa_mdls(fold_str_dir: str, fold_str_file: str, segment_table_dir: str, segment_table_file: str,
                       data_dir: str, fold: int, mdl_opts: dict, sp_cp_dir: Optional[str] = None,
                       ip_cp_dir: Optional[str] = None, neural_gain: float = 10000.0, z_ratio: float = 2.5,
                       subject_filter: Optional[Sequence[int]] = None):
    """
    Synthesizes factor analysis models from neural activity recorded from different fish.

    This function performs basic data pre-processing and then calls synthesize_fa_mdls to do the actual fitting.

    Args:

        fold_str_dir: The directory containing the fold structure file

        fold_str_file: The file containing the fold structures to process

        segment_table_dir: The directory containing the segment table file

        segment_table_file: The segment table file

        data_dir: Directory with the datasets in it

        fold: The fold of data in the fold structures that we are fitting to

        mdl_opts: Dictionary of parameters specifying model structure and fitting to be passed to the function
        synthesize_fa_mdls.  All options can be specified except data, and props (which are determined by
        the subjects fit).  See that function for list of options which must be provided.

        sp_cp_dir: Directory to save check points in for shared posterior fitting.  Can be None, if no check points will
        be created.

        ip_cp_dir: Directory to save check points in for individual posterior fitting.  Can be None, if no check points
        will be created.

        neural_gain: The gain to apply to neural signals.  Applying a gain can be help avoid issues with floating
        point underflow.

        z_ratio: Ratio of z-plane spacing to voxel side length in x-y dimensions

        subject_filter: A list of subject numbers to fit to.  This can be any subset of the subjects in the fold
        structures file. If None, all subjects in the fold structures file will be fit.

    Returns:

        fit_rs: The output of the call to synthesize_fa_mdls.  See that function for more details.

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
                                                                 neural_gain=neural_gain, z_ratio=z_ratio)

    # ======================================================================================================================
    # Extract training data
    # ======================================================================================================================

    print_heading('Forming training data for each subject.')

    train_data = dict()
    for s_n in subjects:
        subject_train_slices = segment_tables[s_n].find_all(fold_groups[s_n][fold]['train'])
        train_data[s_n] = generate_torch_dataset(dataset=subject_data[s_n], slices=subject_train_slices,
                                                 ts_keys=['dff'], inc_time_stamps=False)

    # Convert training data from TimeSeriesBatch objects to Torch tensors
    train_data = {s_n: ts_b.data[0] for s_n, ts_b in train_data.items()}

    # ==================================================================================================================
    # Synthesize models
    # ==================================================================================================================

    print_heading('Synthesizing models.')

    # Specify the ordering of subjects when we go from dictionaries to lists
    subject_order = list(train_data.keys())

    # Add check point directories to fitting options
    for d_i, d in enumerate(mdl_opts['sp_fit_opts']):
        d['cp_save_folder'] = sp_cp_dir
        d['cp_save_str'] = 'rnd_' + str(d_i) + '_'
    for d_i, d in enumerate(mdl_opts['ip_fit_opts']):
        d['cp_save_folder'] = ip_cp_dir
        d['cp_save_str'] = 'rnd_' + str(d_i) + '_'

    fit_rs = synthesize_fa_mdls(data=[train_data[s_n] for s_n in subject_order],
                                props=[subject_neuron_locs[s_n] for s_n in subject_order],
                                **mdl_opts)

    # ======================================================================================================================
    # Return results
    # ======================================================================================================================

    return fit_rs, subject_order
