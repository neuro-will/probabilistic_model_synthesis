""" Basic tools used throughout the project. """

import argparse
import datetime
from typing import List, Union

import torch


def enforce_floor(x: torch.Tensor, flr: float):
    """ Enforces a floor on values in a tensor.

    Args:

        x: The tensor to enforce the floor on

        flr: The floor value

    Returns: Nothing.  The tensor will be modified in place.

    """
    with torch.no_grad():
        x.data[x.data < flr] = flr


def get_lr(optimizer):
    """ Returns the current learning rate of the optimizer. """
    return optimizer.param_groups[0]['lr']


def get_scalar_vl(x: Union[torch.Tensor, float]) -> float:
    """ Helper function to always return either a float no matter the input.

    Args:

        x: The object to get the float from

    Returns:

         flt: The valuea s a float
    """
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().item()
    else:
        return x


def list_to_str(x: List[float]) -> str:
    """ Produces a string of formatted floating point numbers.

    Args:

        x: List of floating point numbers

    Returns:

        s: The formatted string.

    """

    strs = ''
    for x_i in x:
        strs += '{:.2e}, '.format(x_i)

    return strs[0:-2]


def parse_bool_str(s: str):
    """ Parses string input to a boolean value.

    Args:
        s: The string to parse.

    Returns:
        vl: The boolean value.

    Raises:
        ArgumentTypeError: If string value cannot be parsed to a boolean value.
    """
    if s.lower() in ('t', 'true', '1'):
        return True
    elif s.lower() in ('f', 'false', '0'):
        return False
    else:
        raise(argparse.ArgumentTypeError('Boolean value expected.'))


def print_heading(h_str: str):
    """ Prints a heading with string.

    Args:
        h_str: The heading string
    """
    print('============================================================================================')
    time_str = datetime.datetime.today().strftime('%Y-%m-%d, %H:%M:%S: ')
    print(time_str + h_str)
    print('============================================================================================')


def print_info(i_str: str):
    """ Prints infomration.

    Args:
        i_str: The information to print.
    """
    time_str = datetime.datetime.today().strftime('%Y-%m-%d, %H:%M:%S: ')
    print(time_str + i_str)

