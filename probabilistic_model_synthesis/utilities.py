""" Basic tools used throughout the project. """

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


