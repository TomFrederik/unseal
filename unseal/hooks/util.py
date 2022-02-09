from collections import OrderedDict
from typing import Iterable, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

def recursive_module_dict(model: nn.Module) -> OrderedDict:
    """Recursively generates an OrderedDict representing the module structure in a nn.Module.

    :param model: The (sub-)module for which to generate the structure
    :type model: torch.nn.Module
    :return: Structure of the module
    :rtype: OrderedDict
    """
    
    subdict = OrderedDict(module=model, children=OrderedDict())
    if len(model._modules) > 0:
        for name, submodule in model._modules.items():
            subdict['children'][name] = recursive_module_dict(submodule)
    
    return subdict

def create_slice(indices: str) -> slice:
    """Creates a slice object from a string representing the slice.

    :param indices: String representing the slice, e.g. `...,3:5,:`
    :type indices: str
    :return: Slice object corresponding to the input indices.
    :rtype: slice
    """
    if target_shape is not None:
        trailing = len(target_shape) * ",:"
    else:
        trailing = ""
    slice_ = eval(f'np.s_[...,{indices}{trailing}]')
    return slice_


def recursive_to_device(
    iterable: Iterable, 
    device: Union[str, torch.device],
) -> Iterable:
    """Recursively puts an Iterable of (Iterable of (...)) tensors on the given device

    :param iterable: Iterable of tensors or iterables of ...
    :type iterable: Iterable
    :param device: Device on which to put the object
    :type device: Union[str, torch.device]
    :raises TypeError: Unexpected tyes
    :return: Nested iterable with the tensors on the new device
    :rtype: Iterable
    """
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