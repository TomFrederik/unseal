# pre-implemented common hooks
from typing import Iterable, Callable, Optional, Union, List

import torch

from . import util
from .commons import Hook, HookedModel

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

    Example: If the activation has shape ``(B, T, D)`` and replacement tensor has shape ``(D,)`` which you want to plug in
    at position t in the T dimension for every tensor in the batch, then indices should be ``:,t,:``. 

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

def transformers_get_attention(heads: Optional[Union[int, Iterable[int], str]] = None) -> Callable:
    
    # convert string to slice
    if heads is None:
        heads = ":"
    if isinstance(heads, str):
        heads = util.create_slice(heads)

    def func(save_ctx, input, output):
        save_ctx['attn'] = output[2][:,heads,...].detach().cpu()
    
    return func

def gpt_get_attention_hook(layer: int, key: str, heads: Optional[Union[int, Iterable[int], str]] = None) -> Callable:
    func = transformers_get_attention(heads)
    return Hook(f'transformer->h->{layer}->attn', func, key)

def logit_hook(
    layer:int, 
    model: HookedModel, 
    target: Optional[Union[int, List[int]]] = None, 
    position: Optional[Union[int, List[int]]] = None,
    key: Optional[str] = None,
) -> Hook:
    """Create a hook that saves the logits of a layer's output.
    Outputs are saved to save_ctx['{layer}_logits']['logits'].
    
    Currently only works with GPT like models, since it assumes the key of the embedding matrix and the structure of
    these models.

    :param layer: The number of the layer
    :type layer: int
    :param model: The model.
    :type model: HookedModel
    :param target: The target token(s) to extract logits for. Defaults to all tokens.
    :type target: Union[int, List[int]]
    :param position: The position for which to extract logits for. Defaults to all positions.
    :type position: Union[int, List[int]]
    :param key: The key of the hook. Defaults to {layer}_logits.
    :type key: str
    :return: The hook.
    :rtype: Hook
    """
    
    # generate slice
    if target is None:
        target = ":"
    else:
        if isinstance(target, int):
            target = str(target)
        else:
            target = "[" + ",".join(str(t) for t in target) + "]"
    if position is None:
        position = ":"
    else:
        if isinstance(position, int):
            position = str(position)
        else:
            position = "[" + ",".join(str(p) for p in position) + "]"
    position_slice = util.create_slice(f":,{position},:")
    target_slice = util.create_slice(f"{target},:")
    
    # load the relevant part of the vocab matrix
    vocab_matrix = model.structure['children']['transformer']['children']['wte']['module'].weight[target_slice].T
    def inner(save_ctx, input, output):
        save_ctx['logits'] = torch.einsum('bij,jk->bik', output[0][position_slice], vocab_matrix).detach().cpu()
    
    # write key
    if key is None:
        key = str(layer) + '_logits'
    
    # create hook
    hook = Hook(f'transformer->h->{layer}', inner, key)
    
    return hook