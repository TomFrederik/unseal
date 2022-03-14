# Some pre-implemented hooking functions to reproduce the experiments
# from the ROME paper: https://openreview.net/forum?id=mMECu_poAs

# Some hooks that I've only used in the context of the ROME paper so far --> possibly migrate this to the unseal_experiments repo
from typing import Callable, Optional

import torch

from . import util
from . import common_hooks


def additive_output_noise(
    indices: str, 
    mean: Optional[float] = 0, 
    std: Optional[float] = 0.1
) -> Callable:
    slice_ = util.create_slice_from_str(indices)
    def func(save_ctx, input, output):
        noise = mean + std * torch.randn_like(output[slice_])
        output[slice_] += noise
        return output
    return func

def hidden_patch_hook_fn(
    position: int, 
    replacement_tensor: torch.Tensor,
) -> Callable:
    indices = "...," + str(position) + len(replacement_tensor.shape) * ",:"
    inner = common_hooks.replace_activation(indices, replacement_tensor)    
    def func(save_ctx, input, output):
        output[0][...] = inner(save_ctx, input, output[0])
        return output
    return func