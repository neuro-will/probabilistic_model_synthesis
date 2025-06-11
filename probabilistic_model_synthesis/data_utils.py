""" Tools for working with real data.  """

from itertools import chain
from pathlib import Path
from typing import Sequence, List

import numpy as np
import torch

import h5py
import pathlib
import pickle
import scipy.io

from janelia_core.dataprocessing.dataset import PointDataset
from janelia_core.dataprocessing.point import Point
from janelia_core.fileio.data_handlers import NDArrayHandler
from janelia_core.math.basic_functions import copy_and_delay

from probabilistic_model_synthesis.annotations import label_periods
from probabilistic_model_synthesis.annotations import label_subperiods
from probabilistic_model_synthesis.annotations import stim_to_binary_array


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


def load_and_preprocess_data(data_folder: Path, subjects: Sequence[int], stim_vars: Sequence[str] = None,
                             stim_delays: Sequence[int] = None, keep_beh_vars: Sequence[int] = [3,4],
                             normalize_beh_vars: bool = True, neural_gain: float = 10000, beh_gain: float = 100,
                             z_ratio: float = 2.5) -> List[dict]:
    """
    Reads in data as originally distributed on FigShare, converts them to ROIDatasts and applies basic preprocessing.

     The preporcessing consists of:

        1) Down-selecting behavioral variables

        2) Normalizing behavioral signals

        3) Applying gains to the neural data and behavioral signals (potentially useful for avoiding floating point
        issues in later processing)

        4) Extracted selected stimulus variables, representing these in a binary fashion, and forming delayed copies of
        these binary signals

        4) Extracting neuron locations, applying corrections for spacing of z-planes relative to x-y voxel dimensions.

    Args:
        data_folder: Folder holding the original datasets.

        subjects: The ids of the particular subjects to load

        stim_vars: A list of stimulus variables to represent in a binary fashion.  If None, no stimulus
        variables will be represented.

        stim_delays: A list of delays to apply to the stimulus variables.  If None, a value of [0, 1] will be used.

        keep_beh_vars: Indices of behavioral variables to keep.  Indices are of the behavioral variables in the
        generated ROIDatasets.

        normalize_beh_vars: True if behavioral signals should be normalized.  This is done by dividing by the
        max of the kept behavioral signals in the phototaxis periods without shock for each subject.

        neural_gain: The gain to apply to the neural data (dff as well as spikes)

        beh_gain: The gain to apply to the behavioral data

        z_ratio: The ratio of z-plane spacing to voxel x/y dimensions in the original data.

    Returns:

        subject_data: subject_data[n] is the dataset for subject n.  It will have a new ts_data field labeled 'bin_stim'
        for the delayed version of the binary representation of the stimulus.

        neuron_locs: neuron_locs[n] gives the neuron locations for subject n

    """

    if stim_delays is None:
        stim_delays = [0, 1]

    # Load the raw datasets
    n_subjects = len(subjects)
    subject_data = dict()
    for s_i, s_n in enumerate(subjects):
        subject_str = 'subject_' + str(s_n)
        subject_data[s_n] = load_processed_data(Path(data_folder) / subject_str, s_n)
        print('Done loading data for subject ' + subject_str + '.')

    # Down-select behavioral variables
    for s_data in subject_data.values():
        s_data.ts_data['behavior']['vls'] = s_data.ts_data['behavior']['vls'][:, keep_beh_vars]

    # Normalize behavioral signals if we are suppose to
    if normalize_beh_vars:
        for s_n in subjects:
            period_lbls = label_periods(subject_data[s_n].ts_data['stim']['vls'][:])
            period_slices = [d['slice'] for d in period_lbls['phototaxis'] if d['shock'] == False]
            slice_vls = np.concatenate([subject_data[s_n].ts_data['behavior']['vls'][s,:] for s in period_slices], axis=0)
            max_vl = np.max(slice_vls)
            subject_data[s_n].ts_data['behavior']['vls'][:] = subject_data[s_n].ts_data['behavior']['vls'][:]/max_vl

    # Apply gains to data
    for s_n in subjects:
        subject_data[s_n].ts_data['dff']['vls'][:] = neural_gain*subject_data[s_n].ts_data['dff']['vls'][:]
        subject_data[s_n].ts_data['spikes']['vls'][:] = neural_gain*subject_data[s_n].ts_data['spikes']['vls'][:]
        subject_data[s_n].ts_data['behavior']['vls'][:] = beh_gain*subject_data[s_n].ts_data['behavior']['vls'][:]

    # Handle input
    if stim_vars is not None:
        for s_n in subjects:
            stim_n_ts = subject_data[s_n].ts_data['stim']['ts']
            stim_n_vls = subject_data[s_n].ts_data['stim']['vls'][:]
            bin_stim = stim_to_binary_array(stim_n_vls, stim_vars)
            delayed_binary_stim = copy_and_delay(bin_stim, stim_delays)
            subject_data[s_n].ts_data['bin_stim'] = {'ts': stim_n_ts, 'vls': delayed_binary_stim}

    # Pull out neuron locations for each subject
    neuron_locs = dict()
    for s_n in subjects:
        n_subj_neurons = len(subject_data[s_n].point_groups['cells']['points'])
        subj_neuron_locs = np.zeros([n_subj_neurons, 3])
        for n_i in range(n_subj_neurons):
            subj_neuron_locs[n_i,:] = subject_data[s_n].point_groups['cells']['points'][n_i].c[1, :]
        neuron_locs[s_n] = torch.tensor(subj_neuron_locs.astype('float32'))
        neuron_locs[s_n][:, 2] = z_ratio*neuron_locs[s_n][:, 2]

    return [subject_data, neuron_locs]


def load_processed_data(main_folder: str) -> PointDataset:
    """ Loads processed data, as originally distributed on Figshare and then deconvolved.

    Args:
        main_folder: The path to the folder holding the data.  This folder should contain the 'data_full.mat',
        the 'TimeSeries.h5' and 'all_deconvolved.pkl' files for the subject.

    Returns:
        dataset: Object representing the dataset. Structured as follows:
            ts_data:
                ts_data['stim']: Contains stimulus information
                ts_data['behavior']: Contains behavior information
                ts_data['behavior_motor_seed']: Contains behavior motor seed information
                ts_data['dff']: Contains delta F/F for the experiment
                ts_data['calcium']: Contains estimated calcium traces from deconvolution
                ts_data['spikes']: Contains estimated spike trains from deconvolution

            point_groups: Has one group for the cells in the dataset.  Each cell has a raw and registered position.

            metadata:
                metadata['frame_rate'] contains the frame rate

                metadata['deconvolution_params'] is a dictionary with the following fields for parameters of
                the deconvolution for each neuron.  Parameters are listed in the same order as neurons in ts_data.

                    baselines: the baselines of the neurons
                    g: the g parameters of the neurons
                    lam: the lambda parameters of the neurons

            stats:
                stats['mn'] contains the mean image for the experiment.
    """

    main_folder_path = pathlib.Path(main_folder)

    metadata = {}
    ts_data = {}

    # Load the .mat data
    mat_file = main_folder_path / 'data_full.mat'
    mat_data = scipy.io.loadmat(mat_file, struct_as_record=False, squeeze_me=True)
    data = mat_data['data']

    # Load the .h5 data
    h5_file = main_folder_path / 'TimeSeries.h5'
    with h5py.File(h5_file) as f:
        cellResp = f['CellResp'][:]
        absIX = f['absIX'][:]

    # Generate time stamps
    frame_rate = data.fpsec
    metadata['frame_rate'] = frame_rate
    n_time_pts = cellResp.shape[0]
    time_stamps = (1/frame_rate)*np.arange(n_time_pts)

    # Gather stimulus information
    ts_data['stim'] = {'ts': time_stamps,
                       'vls': NDArrayHandler(main_folder, 'stim.pkl', data.stim_full)}

    # Gather behavior information
    ts_data['behavior'] = {'ts': time_stamps,
                           'vls': NDArrayHandler(main_folder, 'behavior.pkl', data.Behavior_full.T)}

    ts_data['behavior_motor_seed'] = {'ts': time_stamps,
                                      'vls': NDArrayHandler(main_folder, 'behavior_motor_seed.pkl',
                                                            data.Behavior_full_motorseed.T)}

    # Gather dff information
    ts_data['dff'] = {'ts': time_stamps,
                      'vls': NDArrayHandler(main_folder, 'dff.pkl', cellResp)}

    # Gather deconvolution information
    deconvolution_file = main_folder_path / 'all_deconvolved.pkl'
    with open(deconvolution_file, 'rb') as f:
        deconvRs = pickle.load(f)

    ts_data['calcium'] = {'ts': time_stamps,
                          'vls': NDArrayHandler(main_folder, 'calcium.pkl', deconvRs['calcium'])}

    ts_data['spikes'] = {'ts': time_stamps,
                         'vls': NDArrayHandler(main_folder, 'spikes.pkl', deconvRs['spikes'])}

    # Gather location of cells
    absIX = np.squeeze(absIX.astype('long'))
    cell_centers = data.CellXYZ
    cell_centers_reg = data.CellXYZ_norm
    n_cells = len(absIX)
    points = [None]*n_cells
    for c_i, abs_ix_i in enumerate(absIX):
        abs_ix_i = abs_ix_i - 1 # Go from MATLAB to Python indexing
        points[c_i] = Point(np.stack([cell_centers[abs_ix_i, :], cell_centers_reg[abs_ix_i, :]]),
                            ['raw', 'reg'])
    cell_points = {'ts_labels': ['dff', 'spikes'], 'points': points}
    point_groups = {'cells': cell_points}

    # Save anatomical information
    stats = {'mn': np.swapaxes(data.anat_stack, 0, 2)}

    # Save deconvolution parameters
    metadata['deconvolution_params'] = dict()
    metadata['deconvolution_params']['baselines'] = deconvRs['baselines']
    metadata['deconvolution_params']['g'] = deconvRs['g']
    metadata['deconvolution_params']['lam'] = deconvRs['lam']

    return PointDataset(ts_data=ts_data, metadata=metadata, point_groups=point_groups, stats=stats)

