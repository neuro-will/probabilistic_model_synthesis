""" Tools for working with real data.  """

from itertools import chain
from pathlib import Path

import numpy as np

from ahrens_wbo.annotations import label_subperiods
from ahrens_wbo.data_processing import load_and_preprocess_data


def read_in_ahrens_data_for_dim_reduction(data_dir: Path, fit_specs: dict, shock: bool, n_validation_slices: int,
                                          preprocess_opts: dict = None, ) -> dict:
    """
    Preprocesses and breaks up Ahrens data into train and validation sets.

    This function also allows the user to request data from different subperiods (e.g., OMR Left, OMR Right) for
    different fish.

    Args:

        data_dir: The directory containing the original datasets

        fit_specs: The keys of this dictionary list integer subjects we want to obtain data for (e.g., 8) and
        values are lists of subperiods we want data from for that subject for (e.g., ['omr_forward', 'omr_left'])

        shock: True if we want data where the shock was applied. False if we want data where the shock was not applied.

        n_validation_slices: The number of slices (each slice corresponds to a single length of time, i.e., trial) that
        should be used for validation for each subperiod.

        pre_process_opts: Options to pass to load_and_proprocess_data (other that data_folder and subjects) for
        preprocessing the data from each subject.  See that function for more details.

    Returns:

        data: A dictionary with the following keys:

            fit_data: fit_data[s_n] is the fitting data for subject s_n, of shape n_smps*n_vars.

            fit_labels: fit_labels[s_n] are integer labels for the fitting data for subject s_n.

            validation_data: validation_data[s_n] is the validation data for subject s_n

            validation_labels: validation_labels[s_n] is the validation labels for subject s_n

        label_map: A dictionary of keys providing string labels for subperiods and values indicating the corresponding
        integer label for that subperiod.

        neuron_locs: neuron_locs[s_n] is the location of neurons (registered to z-brain) for subject s_n, of shape
        n_neurons*3.

    """

    if preprocess_opts is None:
        preprocess_opts = {}

    subjects = list(fit_specs.keys())
    n_subjects = len(subjects)

    datasets, neuron_locs = load_and_preprocess_data(data_folder=data_dir, subjects=subjects, **preprocess_opts)

    # ==================================================================================================================
    # Form the fitting and validation data for each subject
    # ==================================================================================================================

    all_subperiods = set(chain(*[v for v in fit_specs.values()]))
    label_map = {sp: sp_i for sp_i, sp in enumerate(all_subperiods)}

    fit_data = dict()
    validation_data = dict()
    fit_labels = dict()
    validation_labels = dict()

    for s_n, dataset in datasets.items():

        data_n = datasets[s_n].ts_data['dff']['vls'][:]

        # Label the subperiods for this subject
        subperiods = label_subperiods(dataset.ts_data['stim']['vls'][:])

        # Down select to only the subperiods we want to fit on for this subject
        subperiods = {k: v for k, v in subperiods.items() if k in fit_specs[s_n]}

        # Down select to the shock condition we want to fit
        subperiods = {k: [sp_i for sp_i in v if sp_i['shock'] == shock] for k, v in subperiods.items()}

        # Randomly select subperiods for training and validation
        fit_subperiods = dict()
        validation_subperiods = dict()
        for sp_key, sp_slices in subperiods.items():
            n_slices = len(sp_slices)
            validation_inds = np.random.choice(n_slices, n_validation_slices, replace=False)

            validation_slices = [sp_slices[s_i] for s_i in range(n_slices) if s_i in validation_inds]
            fit_slices = [sp_slices[s_i] for s_i in range(n_slices) if s_i not in validation_inds]

            validation_subperiods[sp_key] = validation_slices
            fit_subperiods[sp_key] = fit_slices

        # Pull out the fitting data for this subject
        fit_data[s_n] = {k: np.concatenate([data_n[sl['slice'], :] for sl in v], axis=0)
                         for k, v in fit_subperiods.items()}

        if n_validation_slices > 0:
            validation_data[s_n] = {k: np.concatenate([data_n[sl['slice'], :] for sl in v], axis=0)
                                    for k, v in validation_subperiods.items()}
        else:
            validation_data[s_n] = np.asarray([])

        # Generate numerical labels for each data point
        fit_labels[s_n] = {k: label_map[k] * np.ones(np.sum([sl_i['slice'].stop - sl_i['slice'].start for sl_i in v]))
                           for k, v in fit_subperiods.items()}

        if n_validation_slices > 0:
            validation_labels[s_n] = {k: label_map[k] * np.ones(np.sum([sl_i['slice'].stop - sl_i['slice'].start
                                                                        for sl_i in v]))
                                      for k, v in validation_subperiods.items()}

    # ==================================================================================================================
    # Package and return the data
    # ==================================================================================================================
    fit_data_conc = {k: np.concatenate([data for data in v.values()], axis=0)
                     for k, v in fit_data.items()}
    fit_labels_conc = {k: np.concatenate([lbls for lbls in v.values()], axis=0)
                       for k, v in fit_labels.items()}

    if n_validation_slices > 0:
        validation_data_conc = {k: np.concatenate([data for data in v.values()], axis=0)
                                for k, v in validation_data.items()}
    else:
        validation_data_conc = {k: np.asarray([]) for k in fit_data.keys()}

    if n_validation_slices > 0:
        validation_labels_conc = {k: np.concatenate([lbls for lbls in v.values()], axis=0)
                                  for k, v in validation_labels.items()}
    else:
        validation_labels_conc = {k: np.asarray([]) for k in fit_data.keys()}

    data = {'fit_data': fit_data_conc, 'fit_labels': fit_labels_conc,
            'validation_data': validation_data_conc, 'validation_labels': validation_labels_conc}

    return data, label_map, neuron_locs
