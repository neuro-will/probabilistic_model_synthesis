""" Tools for working with real data.  """

from itertools import chain
from pathlib import Path
from typing import Sequence, List, Callable, Union
import collections

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
from janelia_core.math.basic_functions import combine_slices
from janelia_core.math.basic_functions import find_binary_runs
from janelia_core.math.basic_functions import nan_matrix
from janelia_core.ml.datasets import cat_time_series_batches
from janelia_core.ml.datasets import TimeSeriesDataset

from probabilistic_model_synthesis.annotations import label_periods
from probabilistic_model_synthesis.annotations import label_subperiods
from probabilistic_model_synthesis.annotations import stim_to_binary_array


def break_periods_into_chunks(period_lbls: dict, groups: collections.OrderedDict, chunk_size: int) -> Sequence:
    """ Groups periods together and then breaks these grouped periods into smaller chunks.

    The Ahrens whole brain data consists of different periods in the experiment (e.g, phototaxis, omr, etc...) This
    function can place these periods into groups and then break these grouped periods into smaller contiguous-chunks
    of time.  Samples will be assigned to only one chunk.

    When forming periods into groups, this function distinguishes between periods where shock was and was not delivered,
    and gives the user the option to group these together or place them into separate groups.

    Note that returned chunks will always be contiguous in time.  This means that there may be some data in periods
    which are not assigned to a chunk.

    Args:
        period_lbls: The labeled periods of an experiment, as returned by annotations.label_periods()

        groups: An ordered dictionary specifying which period labels make up a group for the purposes of chunking data.
        groups[i] contains a sequence of dictionaries.  Each of these dictionaries contains two entries.  The 'period'
        entry is a string indicating the label for a period and the 'shock' entry is a binary value indicating if shock
        was delivered in that period or not.

        chunk_size: The size of chunks (in number of samples) that should be formed.

    Returns:
          chunks: chunks[k] contains a list of slice objects indicating the chunks for group k.  If no chunks for
          group k can be formed, chunks[k] will be empty.

    Raises:
        ValueError: If groups is not an OrderedDict

    """

    # Make sure groups is an ordered dict
    if not isinstance(groups, collections.OrderedDict):
        raise(ValueError('groups must be of type collections.OrderedDict'))

    # See what the max time point we need to index is
    max_tm_pt = np.max([np.max([s['slice'].stop for s in period_lbls[k]]) for k in period_lbls]) - 1

    # See what period labels are represented in the current data
    all_lbls = set(period_lbls.keys())

    # Form chunks
    n_groups = len(groups)
    chunks = [None]*n_groups
    for g_i, grp_periods in enumerate(groups.values()):

        # Mark time points in group of periods
        grp_tm_points = np.zeros(max_tm_pt, dtype=bool)
        for group_per_dict in grp_periods:
            group_per = group_per_dict['period']
            group_shock = group_per_dict['shock']
            if group_per in all_lbls:
                period_slices = [d['slice'] for d in period_lbls[group_per] if d['shock'] == group_shock]
                for s in period_slices:
                    grp_tm_points[s] = True

        # Now find contiguous sequences of marked time points
        grp_sequences = find_binary_runs(grp_tm_points)

        # Break each contiguous sequence of marked time points into chunks
        chunk_starts = [np.arange(s.start, s.stop, chunk_size) for s in grp_sequences]
        chunk_starts = [[c_s for c_s in c_starts if c_s + chunk_size <= grp_sequences[seq_i].stop]
                        for seq_i, c_starts in enumerate(chunk_starts)]
        chunk_starts = chain(*chunk_starts)

        chunks[g_i] = [slice(c_start, c_start+chunk_size) for c_start in chunk_starts]

    return chunks


def generate_torch_dataset(dataset: DataSet, slices: Sequence[slice],
                           ts_keys: Sequence[str] = ['dff', 'behavior'],
                           inc_time_stamps: bool = False) -> TimeSeriesBatch:
    """ Constructs a TimeSeriesBatch object from a dataset for training with torch.

    This function has two main purposes. It:

        (1) Allows a user to specify portions of time from a larger dataset to use for training.

        (2) Converts the data to a form that can be used for training with torch.  In particular, it
        will return a TimeSeriesBatch object representing the training data.

    Args:
        dataset: The base dataset object we pull data from.

        slices: A sequence of slice objects we will use in training. Note that each slice specifies the x
        data and the y data will be incremented by 1 from the x data.

        ts_keys: The keys of dataset.ts_data that will will from the dataset from.

        inc_time_stamps: True if time stamps should be returned in the dataset.

    Returns:
        data: The requested data as a TimeSeriesBatch object with the following data entries
            .data[i] will be the data to ts_keys[i]
            .data[-1] will be time stamps if inc_time_stamps are true

    Raises:
        ValueError: If any slice specifies one point or less.
    """

    # Create a TimeSeriesDataset to allow us to easily select our chunks in the sampling format we want
    ts_data = [torch.tensor(dataset.ts_data[k]['vls'][:], dtype=torch.float32) for k in ts_keys]
    if inc_time_stamps:
        ts_data.append(torch.tensor(dataset.ts_data[ts_keys[0]]['ts']))
    dataset = TimeSeriesDataset(ts_data=ts_data)

    # Make sure we have enough samples to form each chunk
    for s in slices:
        if s.stop - s.start < 2:
            raise(ValueError('All slices must contain at least two points.'))

    # Form TimeSeriesBatch object for each slice
    batches = [dataset[s] for s in slices]

    # Put everything together
    return cat_time_series_batches(batches=batches)


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


class SegmentTable:
    """ Holds experiment segment information in a table.

    The abstact model is a table which holds segments in rows and groups of conditions in columns.  The entry in
    the location [s, g] is then the slices for segment s and group g.

    """

    def __init__(self, grp_segment_slices: Sequence[Sequence[Union[Sequence[slice], None]]],
                 segments: Sequence, groups: Sequence):
        """ Creates a new SegmentTable object.

        Args:
            grp_segment_slices: grp_segment_slices[s][g] is a Sequence of slices, indicating the slices for segment
            s and group g or None, indicating there is no data for segment s and group g.

            segments: segements[s] is a labeling for segment s

            groups: groups[g] is an object labeling group g (e.g., this could be a dictionary with information for
            the group)
        """

        self.grp_segment_slices = grp_segment_slices
        self.shape = [len(grp_segment_slices), len(grp_segment_slices[0])]
        self.groups = groups
        self.segments = segments

    def __getitem__(self, item) -> Sequence[Union[Sequence[slice], None]]:
        """ Pulls items from the table.

        Args:
            item: item[0] should be the index for the segment and item[1] should indicate which group(s) of conditions
            to pull

        Returns:
            slices: slices[i] is a Sequence of slices for the requested segment and the i^th requested group condition.
            If there were no slices for the requested segment and group, then slices[i] will be None.

        Raises:
            KeyError: If slices for more than one segment are requested
        """

        n_groups = len(self.grp_segment_slices[0])

        if not(isinstance(item[0], int)):
            raise(KeyError('item[0] must be an integer specifying a single segment'))
        seg_ind = item[0]

        # Code immediately below is to allow us to handle boolean or integer indexing
        group_inds = np.zeros(n_groups, dtype=bool)
        group_inds[item[1]] = True
        group_inds = np.nonzero(group_inds)[0]

        return [self.grp_segment_slices[seg_ind][g_i] for g_i in group_inds]

    def find(self, segment: object, group: object) -> Union[Sequence[slice], None]:
        """ Finds slices for a given segment and group.

        Args:
            segment: The label for the segment to search for

            group: The label for the group to search for

        Returns:

            slices: The slices for the segment and group. If no slices for this segment and group exist, this will
            be None.
        """

        segment_ind = np.argwhere(np.asarray(self.segments) == segment)
        group_ind = np.argwhere(np.asarray(self.groups) == group)

        if len(segment_ind) == 0:
            raise(KeyError('Segment ' + str(segment) + ' does not exist.'))
        else:
            segment_ind = segment_ind[0][0]

        if len(group_ind) == 0:
            raise(KeyError('Group ' + str(group) + ' does not exist.'))
        else:
            group_ind = group_ind[0][0]

        return self.grp_segment_slices[segment_ind][group_ind]

    def find_all(self, grp_dict: dict) -> Union[Sequence[slice], None]:
        """ Finds all slices in all requested segments for one or more groups.

    Args:
        grp_dict: A dictionary with keys specifying groups and values which are a list of segments to include
        for that group.

    Returns:

        slices: The requested slices, which will be combined.
    """

        groups = grp_dict.keys()
        slices = []
        for group in groups:
            group_segments = grp_dict[group]
            for segment in group_segments:
                slices.append(self.find(segment=segment, group=group))

        slices = [sl for sl in slices if sl is not None]
        return combine_slices(list(chain(*slices)))

    def n_group_segments(self, group: str) -> int:
        """ Returns the number of segments that exist for a group. """

        cnt = 0
        for segment in self.segments:
            if self.find(segment=segment, group=group) is not None:
                cnt += 1
        return cnt

    def to_dict(self):
        """ Converts a segment table to a dictionary for serialization.

        Returns:
            A dictionary for the table.
        """
        return vars(self)

    @classmethod
    def from_dict(cls, d: dict):
        """ Creates a SegmentTable from a dictionary.

        Args:
            d: The dictionary to create the table from

        Returns:
            table: The created table

        """

        return cls(grp_segment_slices = d['grp_segment_slices'],
                   segments=d['segments'],
                   groups=d['groups'])


def segment_dataset(period_lbls: dict, groups: collections.OrderedDict, chunk_size: int, segment_labels: Sequence,
                    segment_ratios: Sequence[int], vls: np.ndarray = None,
                    vl_fnc: Callable[[np.ndarray], float] = None, random_vl_assignment: bool = True) -> SegmentTable:
    """ Segments experimental data into disjoint sets.

    This function:

        (1) Allows a user to specify groups of periods of an experiment (e.g., 'omr' w/o shock + 'looming'
            w/o shock could be a group).

        (2) Will find contiguous periods of time in the experiment when the experiment is in the condition(s)
            for each group and break up each of these contiguous periods into a number of "chunks" of time (each
            chunk is contiguous in time).

        (3) Will then assign data from each group to each segment.  The percentage of data assigned to each segment for
        a group is equal to the percent of total data the user assigns that segment to have.

    When assigning chunks of data to segments, chunks can be randomly assigned OR they can be ordered by value (see
    options below) and then assigned so that segments are also roughly balanced proportionally in value.

    Args:

        period_lbls: The labeled periods of an experiment, as returned by annotations.label_periods()

        groups: A dictionary specifying groupings of experimental periods.  The key for each entry gives a name we
        want to refer to a group by.  The value for each entry is a sequence of normal dictionaries. Each of these
        dictionaries contains two entries: The 'period' entry is a string indicating the label for a period and
        the 'shock' entry is a binary value indicating if shock was delivered in that period or
        not.

        chunk_size: The size of chunks that will be formed.

        segment_labels: Labels for each segment

        segment_ratios: Specifies how much data is assigned to each segment.  Segment i will have a percentage of
        data equal to segment_ratios[i]/sum(segment_ratios) assigned to it.  All entries in segment_ratios must
        be integers.

        vls: A scalar value to associate with each sample point in the experiment.  If provided, these values will be
        used to order chunks when assigning them to segments.  The function vl_func takes all the scalar values in a
        chunk and assigns them a value.  If vls is None, then chunks are randomly assigned to segments.

        vl_fnc: A function for taking all the scalar values in a chunk and assigning them a value.  This function
        should accept a 1-d numpy array and return a scalar.  This function must be provided if vls is not None.

        random_vl_assignment: When assigning chunks to segments by value, after ordering the chunks, we can determine
        how we assign each block of chunks to segmemts.  If this value is false, then chunks are sorted and assigned
        to segments sequentially.  For example, let's say segment_ratios is [3, 2].  If random_vl_assignment is false,
        then the chunks with the top 3 values will be assigned to the first segment and then chunks with the next two
        largest values will be assigned to the second segment.  While this may have advantages in some small number of
        cases, it will produce a bias where the second segment has smaller values than the first.  If this value is
        true, then assignment is randomized, so using the example introduced, the first 5 values are randomly assigned
        to blocks 1 and 2, then the next five values would be randomly assigned to the segments, and so on.

    Returns:

        segmentTable: The table with the requested segmentation. segmentTable[s, g] contains slices for the segment
        corresponding to segment_ratios[s] and group groups[g].  The group labels for the segment table will
        be the keys of the groups input.

    Raises:
        ValueError: If segment_ratios is not a sequence
        ValueError: If segment_ratios is not a sequence of ints

    """

    # Do checks on input
    if not hasattr(segment_ratios, '__iter__'):
        raise(TypeError('segment_ratios must be a sequence'))
    if not np.all([np.issubdtype(type(vl), np.integer) for vl in segment_ratios]):
            raise(TypeError('segment ratios must be a sequence of ints'))

    n_grps = len(groups)
    n_segments = len(segment_ratios)

    # Break up data into contiguous chunks for each group
    group_chunks = break_periods_into_chunks(period_lbls=period_lbls, groups=groups, chunk_size=chunk_size)

    # Order the chunks in each group
    if vls is None:
        # If no values are provided, we randomly order chunks in each group
        for g_i in range(n_grps):
            n_chunks = len(group_chunks[g_i])
            group_chunks[g_i] = [group_chunks[g_i][j] for j in np.random.permutation(n_chunks)]
    else:
        # If values are provided, we sort chunks by the values
        for g_i in range(n_grps):
            chunk_vls = np.asarray([vl_fnc(vls[sl]) for sl in group_chunks[g_i]])
            chunk_sort_order = np.flip(np.argsort(chunk_vls)) # Values are sorted in descending order
            group_chunks[g_i] = [group_chunks[g_i][j] for j in chunk_sort_order]

    # Now we segment the data in each group
    ratio_norm = np.sum(segment_ratios)
    base_block = np.concatenate([seg_i*np.ones(n_s) for seg_i, n_s in enumerate(segment_ratios)])
    grp_segment_slices = [None]*n_grps
    for g_i in range(n_grps):
        n_chunks = len(group_chunks[g_i])
        n_blocks = n_chunks//ratio_norm # See how many whole blocks we can form

        # Randomly assign chunks in each block to segments, respecting proportions specified in segment_ratios
        segment_assignments = nan_matrix(n_chunks)
        for block_i in range(n_blocks):
            start_ind = block_i*ratio_norm
            end_ind = start_ind + ratio_norm
            if random_vl_assignment:
                segment_assignments[start_ind:end_ind] = base_block[np.random.permutation(ratio_norm)]
            else:
                segment_assignments[start_ind:end_ind] = base_block

        # Pull out the slices for each segment for this group
        segment_slices = [None]*n_segments
        for seg_i in range(n_segments):
            segment_inds = np.squeeze(np.argwhere(segment_assignments==seg_i))
            segment_slices[seg_i] = [group_chunks[g_i][i] for i in segment_inds]
        grp_segment_slices[g_i] = segment_slices

    # Transpose grp_segment_slices
    grp_segment_slices = [[grp_segment_slices[grp_i][seg_i] for grp_i in range(n_grps)]
                           for seg_i in range(n_segments)]

    return SegmentTable(grp_segment_slices=grp_segment_slices, segments=segment_labels, groups=list(groups.keys()))


def segment_dataset_with_constant_segment_sizes(period_lbls: dict, groups: collections.OrderedDict, chunk_size: int,
                                                n_segment_chunks: int, vls: np.ndarray = None,
                                                vl_fnc: Callable[[np.ndarray], float] = None,
                                                random_vl_assignment: bool = True) -> SegmentTable:
    """ Segments experimental data into disjoint sets.

    This function:

        (1) Allows a user to specify groups of periods of an experiment (e.g., 'omr' w/o shock + 'looming'
            w/o shock could be a group).

        (2) Will find contiguous periods of time in the experiment when the experiment is in the condition(s)
            for each group and break up each of these contiguous periods into a number of "chunks" of time (each
            chunk is contiguous in time).

        (3) Will then assign data from each group to each segment.  The user specifies how many chunks to assign
        to each segment.  The number of segments created will be determined by the amount of data available.

    When assigning chunks of data to segments, chunks can be randomly assigned OR they can be ordered by value (see
    options below) and then assigned so that segments are also roughly balanced proportionally in value.

    Args:

        period_lbls: The labeled periods of an experiment, as returned by annotations.label_periods()

        groups: A dictionary specifying groupings of experimental periods.  The key for each entry gives a name we
        want to refer to a group by.  The value for each entry is a sequence of normal dictionaries. Each of these
        dictionaries contains two entries: The 'period' entry is a string indicating the label for a period and
        the 'shock' entry is a binary value indicating if shock was delivered in that period or
        not.

        chunk_size: The size of chunks that will be formed.

        n_segment_chunks: The number of chunks to assign to each segment.

        vls: A scalar value to associate with each sample point in the experiment.  If provided, these values will be
        used to order chunks when assigning them to segments.  The function vl_func takes all the scalar values in a
        chunk and assigns them a value.  If vls is None, then chunks are randomly assigned to segments.

        vl_fnc: A function for taking all the scalar values in a chunk and assigning them a value.  This function
        should accept a 1-d numpy array and return a scalar.  This function must be provided if vls is not None.

        random_vl_assignment: When assigning chunks to segments by value, after ordering the chunks, we can determine
        how we assign each block of chunks to segmemts.  If this value is false, then chunks are sorted and assigned
        to segments sequentially.  While this may have advantages in some small number of cases, it will produce a bias
        where the second segment has smaller values than the first.  If this value is true, then assignment is
        randomized to remove this bias.

    Returns:

        segmentTable: The table with the requested segmentation. segmentTable[s, g] contains slices for the segment
        corresponding to segment_ratios[s] and group groups[g].  The group labels for the segment table will
        be the keys of the groups input.  Note that segmentTables[s, g] may have no slices, since different groups
        may have different amounts of data (so we may not be able to assign data for all segments for all groups).

    """
    n_grps = len(groups)

    # Break up data into contiguous chunks for each group
    group_chunks = break_periods_into_chunks(period_lbls=period_lbls, groups=groups, chunk_size=chunk_size)

    # Order the chunks in each group
    if vls is None:
        # If no values are provided, we randomly order chunks in each group
        for g_i in range(n_grps):
            n_chunks = len(group_chunks[g_i])
            group_chunks[g_i] = [group_chunks[g_i][j] for j in np.random.permutation(n_chunks)]
    else:
        # If values are provided, we sort chunks by the values
        for g_i in range(n_grps):
            chunk_vls = np.asarray([vl_fnc(vls[sl]) for sl in group_chunks[g_i]])
            chunk_sort_order = np.flip(np.argsort(chunk_vls)) # Values are sorted in descending order
            group_chunks[g_i] = [group_chunks[g_i][j] for j in chunk_sort_order]

    # Now we segment the data in each group
    grp_segment_slices = [None]*n_grps
    for g_i, chunks_i in enumerate(group_chunks):
        # Pull out the slices for each segment for this group
        n_chunks_i = len(chunks_i)
        n_segments_i = n_chunks_i // n_segment_chunks

        # Assign chunks to segments
        base_block_i = np.arange(n_segments_i)
        segment_assignments_i = nan_matrix(n_chunks_i)
        for b_i in range(n_segment_chunks):
            block_start_ind = b_i*n_segments_i
            block_end_ind = block_start_ind + n_segments_i
            if random_vl_assignment:
                segment_assignments_i[block_start_ind:block_end_ind] = base_block_i[np.random.permutation(n_segments_i)]
            else:
                segment_assignments_i[block_start_ind:block_end_ind] = base_block_i

        # Pull out the assigned slices for each segment here
        segment_slices_i = [None]*n_segments_i
        for seg_i in range(n_segments_i):
            segment_inds_i = np.squeeze(np.argwhere(segment_assignments_i == seg_i), axis=1)
            segment_slices_i[seg_i] = [chunks_i[c_i] for c_i in segment_inds_i]
        grp_segment_slices[g_i] = segment_slices_i

    # Transpose grp_segment_slices and add None values where needed
    max_n_segments = np.max([len(vls) for vls in grp_segment_slices])
    new_grp_segment_slices = [None]*max_n_segments
    for s_i in range(max_n_segments):
        new_grp_segment_slices[s_i] = [None]*n_grps
    for g_i in range(n_grps):
        for s_i, slices in enumerate(grp_segment_slices[g_i]):
            new_grp_segment_slices[s_i][g_i] = slices

    segment_labels = ['set_' + str(i) for i in range(max_n_segments)]

    return SegmentTable(grp_segment_slices=new_grp_segment_slices, segments=segment_labels, groups=list(groups.keys()))

