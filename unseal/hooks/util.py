from collections import OrderedDict
from typing import Iterable, Tuple, Union, List

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
) -> Union[List, Tuple, torch.Tensor]:
    """Recursively puts an Iterable of (Iterable of (...)) tensors on the given device
    This creates a copy of the iterable in question, which can be very costly.

    :param iterable: Tensor or Iterable of tensors or iterables of ...
    :type iterable: Tensor or Iterable
    :param device: Device on which to put the object
    :type device: Union[str, torch.device]
    :raises TypeError: Unsupported type.
    :return: Nested iterable with the tensors on the new device. Default type is List, unless iterable is a tuple in which case a tuple is returned.
    :rtype: Union[List, Tuple, torch.Tensor]
    """
    if isinstance(iterable, torch.Tensor):
        return iterable.to(device)
    elif isinstance(iterable, Iterable):
        out = [recursive_to_device(o, device) for o in iterable]
        if isinstance(iterable, Tuple):
            out = tuple(out)
        return out
    else:
        raise TypeError(f"Can not handle object of type {type(iterable)}.")

def recursive_detach(
    iterable: Union[torch.Tensor, Iterable]
) -> Union[torch.Tensor, List, Tuple]:
    """Recursively detaches an Iterable of (Iterable of (...)) tensors.
    This creates a copy of the iterable in question, which can be very costly.

    :param iterable: Tensor or Iterable of tensors or iterables of ...
    :type iterable: Union[torch.Tensor, Iterable]
    :raises TypeError: Unsupported type.
    :return: Nested iterable with the tensors detached. Default type is List, unless iterable is a tuple in which case a tuple is returned.
    :rtype: Union[torch.Tensor, List, Tuple]
    """
    if isinstance(iterable, torch.Tensor):
        return iterable.detach()
    elif isinstance(iterable, Iterable):
        out = [recursive_detach(o) for o in iterable]
        if isinstance(iterable, tuple):
            out = tuple(out)
        return out
    else:
        raise TypeError(f"Can not detach object of type {type(iterable)}.")