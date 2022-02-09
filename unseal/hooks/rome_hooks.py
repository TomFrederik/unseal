# Some pre-implemented hooking functions to reproduce the experiments
# from the ROME paper: https://openreview.net/forum?id=mMECu_poAs

from typing import Callable

import torch

from . import util


# general replace activation function
def replace_activation(indices: str, replacement_tensor: torch.Tensor) -> Callable:
    """Replaces activation with replacement tensor. Indices are filled from back to front

    Example: If the activation has shape (B, T, D) and replacement tensor has shape (D,) which you want to plug in
    at position t in the T dimension for every tensor in the batch, then indices should be "t". 

    :param indices: Indices at which to insert the replacement tensor, excluding the shape of the replacement tensor itself.
    :type indices: str
    :param replacement_tensor: Tensor that is filled in.
    :type replacement_tensor: torch.Tensor
    :return: Function that replaces part of a given tensor with replacement_tensor
    :rtype: Callable
    """
    slice_ = util.create_slice(indices, replacement_tensor.shape)
    def func(save_ctx, input, output):
        # add dummy dimensions if shape mismatch
        diff = len(output[slice_].shape) - len(replacement_tensor.shape)
        rep = replacement_tensor
        for i in range(diff):
            rep = rep[None]
        # replace part of tensor    
        output[slice_] = rep.to(output.device)
        return output

    return func

def additive_noise(indices, mean=0, std=0.1):
    slice_ = util.create_slice(indices)
    def func(save_ctx, input, output):
        noise = mean + std * torch.randn_like(output[slice_])
        output[slice_] += noise
        return output
    return func

# special to ROME reimplementation
def hidden_patch_hook_fn(position, replacement_tensor):
    inner = replace_activation(str(position), replacement_tensor)    
    def func(save_ctx, input, output):
        output[0][...] = inner(save_ctx, input, output[0])
        return output
    return func