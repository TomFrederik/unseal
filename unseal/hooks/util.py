from collections import OrderedDict
from typing import Iterable, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

def create_slice_from_str(indices: str) -> slice:
    """Creates a slice object from a string representing the slice.

    :param indices: String representing the slice, e.g. ``...,3:5,:``
    :type indices: str
    :return: Slice object corresponding to the input indices.
    :rtype: slice
    """
    if len(indices) == 0:
        raise ValueError('Empty string is not a valid slice.')
    return eval(f'np.s_[{indices}]')


def recursive_to_device(
    iterable: Union[Iterable, torch.Tensor], 
    device: Union[str, torch.device],
) -> Iterable:
    """Recursively puts an Iterable of (Iterable of (...)) tensors on the given device

    :param iterable: Tensor or Iterable of tensors or iterables of ...
    :type iterable: Tensor or Iterable
    :param device: Device on which to put the object
    :type device: Union[str, torch.device]
    :raises TypeError: Unexpected tyes
    :return: Nested iterable with the tensors on the new device
    :rtype: Iterable
    """
    if isinstance(iterable, torch.Tensor):
        return iterable.to(device)

    new = []
    for i, item in enumerate(iterable):
        if isinstance(item, torch.Tensor):
            new.append(item.to(device))
        elif isinstance(item, Iterable):
            new.append(recursive_to_device(item, device))
        else:
            raise TypeError(f'Expected type tensor or Iterable but got {type(item)}.')
    if isinstance(iterable, Tuple):
        new = tuple(new)
    return new