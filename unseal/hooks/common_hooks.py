# pre-implemented common hooks
from typing import Iterable

import torch

from . import util

def save_output(save_ctx: dict, input: torch.Tensor, output: torch.Tensor):
    """Basic hooking function for saving the output of a module to the global context object

    :param save_ctx: Context object
    :type save_ctx: dict
    :param input: Input to the module.
    :type input: torch.Tensor
    :param output: Output of the module.
    :type output: torch.Tensor
    """
    if isinstance(output, torch.Tensor):
        save_ctx['output'] = output.to('cpu')
    elif isinstance(output, Iterable): # hope for the best
        save_ctx['output'] = util.recursive_to_device(output, 'cpu')


def replace_activation(indices: str, replacement_tensor: torch.Tensor) -> Callable:
    """Creates a hook which replaces a module's activation (output) with a replacement tensor. 
    If there is a dimension mismatch, the replacement tensor is copied along the leading dimensions of the output.

    Example: If the activation has shape (B, T, D) and replacement tensor has shape (D,) which you want to plug in
    at position t in the T dimension for every tensor in the batch, then indices should be ":,t,:". 

    :param indices: Indices at which to insert the replacement tensor
    :type indices: str
    :param replacement_tensor: Tensor that is filled in.
    :type replacement_tensor: torch.Tensor
    :return: Function that replaces part of a given tensor with replacement_tensor
    :rtype: Callable
    """
    slice_ = util.create_slice(indices)
    def func(save_ctx, input, output):
        # add dummy dimensions if shape mismatch
        diff = len(output[slice_].shape) - len(replacement_tensor.shape)
        rep = replacement_tensor[(None for _ in range(diff))].to(output.device)
        # replace part of tensor    
        output[slice_] = rep
        return output

    return func