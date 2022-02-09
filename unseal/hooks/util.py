from collections import OrderedDict
from typing import Iterable, Tuple

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

def create_slice(indices, target_shape=None):
    if target_shape is not None:
        trailing = len(target_shape) * ",:"
    else:
        trailing = ""
    slice_ = eval(f'np.s_[...,{indices}{trailing}]')
    return slice_


def recursive_to_device(iterable, device):
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