""" Contains objects and functions useful for annotating whole-brain organization datasets. """

from collections import OrderedDict
import copy
import re
from typing import Sequence

import numpy as np

from janelia_core.math.basic_functions import find_binary_runs

# Define a dictionary which defines what each column of ts_data['behavior']['vls'] corresponds to in Ahrens
# whole-brain organization datasets.
beh_dict = OrderedDict([('left_swims', 0),
                        ('right_swims', 1),
                        ('forward_swims', 2),
                        ('left_channel', 3),
                        ('right_channel', 4)])

# Define a dictionary which defines which values in ts_data['stim']['vls'] correspond to

stim_dict = OrderedDict([('black', 0),
                         ('phototaxis_right', 1),
                         ('phototaxis_left', 2),
                         ('white', 3),
                         ('grey', 4),
                         ('omr_backward', 9),
                         ('omr_forward', 10),
                         ('omr_right', 11),
                         ('omr_left', 12),
                         ('dot', 13),
                         ('looming_left', 14),
                         ('looming_right', 15),
                         ('shock', 16),
                         ('red_blue_right', 21),
                         ('red_blue_left', 22),
                         ('red_red', 23)])


def stim_to_binary_array(stim: np.ndarray, stim_conds: Sequence[str]) -> np.ndarray:
    """ Converts a 1-d array of stimulus values to a binary matrix representation.

    Args:
        stim: The 1-d array of stimulus values to convert.

        stim_conds: stim_conds[i] contains a string for the stimulus condition that should be represented in the i^th
        column of the binary matrix.  String conditions must be contained in the stim_dict of this module.

    Returns:
          bin_stim: an array of shape t*n_s, where t is the number of time points represented in stim and n_s is the
          length of stim_conds.  bin_stim[j, i] is 1 if stim_conds[i] was present at time point j.

    Raises:
        ValueError: If one or more entries in stim_cond are not valid stimulus strings (that is they are not in the
        stim_dict of this module.)
    """

    # Make sure all requested stim conditions are valid
    if not (set(stim_conds) <= set(stim_dict.keys())):
        raise(ValueError('stim_conds contains one or more conditions not in annotations.stim_dict'))

    # Produce the binary representation
    n_tm_pts = len(stim)
    n_stim_conds = len(stim_conds)
    bin_stim = np.zeros([n_tm_pts, n_stim_conds], dtype='bool')

    for c_i, s_cond in enumerate(stim_conds):
        s_inds = (stim == stim_dict[s_cond])
        bin_stim[s_inds, c_i] = True

    return bin_stim


def label_periods(stim: np.ndarray, min_l: int = 200) -> dict:
    """ Labels periods of time in an Ahrens Whole-Brain Imaging experiment.

    The wbo data contains a stimulus value that indicates at each moment in time what stimulus was presented to the
    animal.  E.g., black, white, phototaxis left, etc... These stimuli were presnted in blocks (so there were whole
    blocks of alternating phototaxis stimuli, for example).  This function uses the stim variable of an experiment to
    find and label these larger blocks.

    Args:
        stim: The stimulus information, of length T, where T is the number of time points in the dataset.  stim[t]
        contains an integer value indicating what stimulus was present at time t.

        min_l: The minimum number of sequential time points stimuli for a period need to be present to be marked as
        a period.  Because this function searches for different types in stimulus events in series, (it searches for
        flash before phototaxis) and it applies a simple "OR" condition to group stimulus events into periods, this
        value should be set reasonable large to avoid mis-identifying something like a black period in phototaxis as
        part of a flash period.

    Returns:
        periods: A dictionary with keys for each period in an experiment. Each entry will contain a list, with each
        entry of the list containing another dictionary specific to each period in time.  The dictionary for each
        period in time will have a 'slice' field, indicating the contiguous time points for that period and a
        'shock' field which will be 1 if shocks were delivered during this period and 0 otherwise.
        For example, periods['phototaxis'][0] will contain a dictionary for the first period of phototaxis stimuli
        in an experiment.  If a period was not present in an experiment, no key for that period will be present in
        periods.

    Raises:
        RuntimeError: If an internal logic check fails to find contiguous periods of the experiment.

    """

    # Note: We convert the stimulus array to a byte representation, allowing us to more easily use regexp

    # Get the byte values of different stimulus labels
    stim_dict_b = copy.deepcopy(stim_dict)
    for k in stim_dict_b.keys():
        stim_dict_b[k] = stim_dict_b[k].to_bytes(1, 'big')

    # Convert stimulus variable to byte array
    stim_ba = bytearray([s_i for s_i in stim])

    # Create the dictionary of regular expressions we will search for - the order we enter them in the dictionary
    # corresponds to their precedence

    re_dict = OrderedDict()

    re_dict['flash'] = (b'[' + stim_dict_b['black'] + stim_dict_b['white'] +
                        stim_dict_b['shock'] + b']' + b'+')

    re_dict['phototaxis'] = (b'[' + stim_dict_b['white'] + stim_dict_b['black'] + stim_dict_b['phototaxis_right'] +
                             stim_dict_b['phototaxis_left'] + stim_dict_b['shock'] + b']' + b'+')

    re_dict['omr'] = (b'[' + stim_dict_b['white'] + stim_dict_b['omr_backward'] + stim_dict_b['omr_forward'] +
                      stim_dict_b['omr_right'] + stim_dict_b['omr_left'] + stim_dict_b['shock'] + b']' + b'+')

    re_dict['looming'] = (b'[' + stim_dict_b['white'] + stim_dict_b['looming_left'] +
                             stim_dict_b['looming_right'] + stim_dict_b['shock'] + b']' + b'+')

    re_dict['spontaneous'] = (b'[' + stim_dict_b['grey'] + stim_dict_b['shock'] + b']' + b'+')

    re_dict['dot'] = (b'[' + stim_dict_b['black'] + stim_dict_b['dot'] + stim_dict_b['shock'] + b']' + b'+')

    re_dict['colors'] = (b'[' + stim_dict_b['red_blue_right'] + stim_dict_b['red_blue_left'] +
                         stim_dict_b['red_red'] + stim_dict_b['shock'] + b']' + b'+')

    # Label time points, respecting the precedence in the dictionary
    periods = dict()
    n_tm_pts = len(stim)
    valid_pts = np.ones(n_tm_pts, dtype=bool)
    for lbl, regexp in re_dict.items():
        matches = re.finditer(regexp, stim_ba)
        lbl_periods = []
        for m in matches:
            match_valid_pts = np.zeros(n_tm_pts, dtype=bool)
            match_valid_pts[m.span()[0]:m.span()[1]] = True
            match_valid_pts = np.logical_and(valid_pts, match_valid_pts)
            match_valid_inds = np.nonzero(match_valid_pts == True)[0]

            if len(match_valid_inds) > 0: # Make sure we have at least some valid points
                first_ind = match_valid_inds[0]
                last_ind = match_valid_inds[-1]
                match_len = last_ind - first_ind + 1

                # Do a sanity check to make sure the period is contiguous
                if not np.all(match_valid_pts[first_ind:last_ind]):
                    raise(RuntimeError('Caught discontiguous period.'))

                if match_len >= min_l: # Only proceed if the length of the period is above minimum
                    match_slice = slice(first_ind, last_ind + 1)
                    match_shock = np.any(stim[match_slice] == stim_dict['shock'])
                    period_dict = {'slice': match_slice, 'shock': match_shock}
                    lbl_periods.append(period_dict)
                    valid_pts[match_slice] = False
        if len(lbl_periods) > 0:
            periods[lbl] = lbl_periods

    return periods


def label_subperiods(stim: np.ndarray, min_l: int = 20) -> dict:
    """ Labels subperiods of time in an Ahrens whole-brain imaging experiment.

    A subperiod is a period plus the defining stimulus (e.g., phototaxis left).

    Args:
        stim: The stimulus information, of length T, where T is the number of time points in the dataset.  stim[t]
        contains an integer value indicating what stimulus was present at time t.

        min_l: The minimum number of sequential time points stimuli for a period need to be present to be marked as
        a subperiod.

    Returns:
        subperiods: A dictionary with keys for each subperiod in an experiment. Each entry will contain a list, with
        each entry of the list containing another dictionary specific to each subperiod in time.  The dictionary for
        each subperiod in time will have a 'slice' field, indicating the contiguous time points for that subperiod and a
        'shock' field which will be 1 if shocks were delivered during this subperiod and 0 otherwise.  If a subperiod
        is not present in an experiment, no key for that subperiod will be present in subperiods.
    """

    # Do basic labeling of periods so we can see where shock is present
    period_labels = label_periods(stim)
    shock_times = np.zeros_like(stim)
    for period in period_labels.keys():
        for block in period_labels[period]:
            if block['shock']:
                shock_times[block['slice']] = True

    # Make a copy of the stim dict without the shock annotation
    stim_dict_wo_shock = copy.deepcopy(stim_dict)
    del stim_dict_wo_shock['shock']

    # Label subperiods
    subperiods = dict()

    for subperiod in stim_dict_wo_shock.keys():
        match_inds = stim == stim_dict[subperiod]
        match_runs = find_binary_runs(match_inds)

        # Only keep runs which are long enough
        keep_runs = [run for run in match_runs if (run.stop - run.start) >= min_l]

        # See which runs have a shock in them
        n_keep_runs = len(keep_runs)
        if n_keep_runs > 0:
            runs_w_shock = [None]*n_keep_runs
            for r_i, run in enumerate(keep_runs):
                runs_w_shock[r_i] = np.any(shock_times[run])

            # Generate annotations
            subperiods[subperiod] = [{'slice': run, 'shock': has_shock} for run, has_shock in zip(keep_runs, runs_w_shock)]

    return subperiods

