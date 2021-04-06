""" Contains basic math and stats functions. """

from typing import Callable

import numpy as np
import torch


class MeanFcnTransformer(torch.nn.Module):
    """ A module which applies a non-learnable learning transform to mean vectors.

    In particular, if x is a vector of random variables and we apply the transform y = o*x,
    this module will compute the mean of y.

    This module assumes x is itself computed from some function f, which is also provided
    to this module.  This module is then helpful when wanting to transform the output
    of a module computing means by a fixed linear transform.

    """

    def __init__(self, o: np.ndarray, f: torch.nn.Module):
        """ Creates a new MeanFcnTransformer object.

        Args:
            o: The transform to apply to the means.

            f: A function which generates means given input.  Should accept a matrix where each row is an input and
            output a matrix where each row is the computed mean for the input.
        """
        super().__init__()
        self.f = f
        self.o = torch.tensor(o)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Computes input from outoput.

        Args:
            x: Input - each row represents a mean vector

        Returns:
            y: Output - each row is a transformed mean vector
        """
        x = self.f(x)
        print(x.shape)
        return torch.matmul(x, self.o.t())


class StdFcnTransformer(torch.nn.Module):
    """ A modules that calculates the result on standard deviations of applying a linear transform to random variables.

    In particular, if x is a vector of random variables and we apply the transform y = o*x,
    this module will compute the standard deviation of the variables in y, given the standard deviations of
    the variables in x.

    This module assumes x is itself computed from some function f, which is also provided
    to this module.  This module is then helpful when wanting to transform the output
    of a module computing means by a fixed linear transform.

    *** NOTE *** The calculations implemented here will only be correct if the variables in x are not correlated.

     """

    def __init__(self, o: np.ndarray, f: torch.nn.Module):
        """ Creates a new StdFcnTransformer object.

        Args:

            o: The transform to apply

            f: A function which generates standard deviations given input.  Should accept rows of input and give
            standard deviations (as rows of a new matrix) for each input row.
        """
        super().__init__()
        self.f = f
        self.o = torch.tensor(o)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Computes input from output.

        Args:
            x: Input - each row gives the standard deviation of random variables

        Returns:
            y: Output - each row gives the standard deviation of the transformed variables

        """
        x = self.f(x)
        return torch.sqrt(torch.matmul(x**2, (self.o**2).t()))


