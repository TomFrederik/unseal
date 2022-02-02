from collections import OrderedDict
from functools import partial
from itertools import chain
import logging
from typing import Callable, Optional, Union, List, Tuple

import numpy as np
import torch

class ModuleHook:
    """Hook class to save features computed during forward pass of given module
    """
    def __init__(
        self, 
        module: torch.nn.Module, 
        output_fn: Optional[Callable] = None
    ) -> None:
        """Constructor for ModuleHook

        :param module: Module to which the hook should be attached. In practice, this is usually a layer within a larger Module.
        :type module: torch.nn.Module
        :param output_fn: function which is applied to the output, defaults to None
        :type output_fn: Optional[Callable]
        :return: None
        """
        self.hook = module.register_forward_hook(partial(self.hook_fn, output_fn=output_fn))
        self.module = None
        self.features = None

    def hook_fn(
        self, 
        module: torch.nn.Module, 
        input: torch.Tensor, 
        output: torch.Tensor,
        output_fn: Optional[Callable] = None,
    ) -> None:
        """Hooking function that stores the hooked module's output (its features)

        :param module: hooked module
        :type module: torch.nn.Module
        :param input: inpute to module
        :type input: torch.Tensor
        :param output: output of module on given input
        :type output: torch.Tensor
        :param output_fn: function which is applied to the output, defaults to None
        :type output_fn: Optional[Callable]
        """
        self.module = module

        # modify output
        if output_fn is None:
            output_fn = lambda x: x
        
        output = output_fn(output)
        self.features = output
        
        return output


    def close(self):
        self.hook.remove()


class FullModelHooks:
    def __init__(self, model):
        """Object that stores all hooks for a model. Initialized with 'basic' hooks which only give the output of each layer.
        Retrieve hooks via indexing or calling this object. Add new hooks via `add_custom_hook()`

        :param model: Model to be hooked
        :type model: nn.Module
        :raises TypeError: Incorrect model type
        """
        # check inputs
        if not isinstance(model, torch.nn.Module):
            raise TypeError(f"model should be type torch.nn.Module but is {type(model)}")
        
        self.model = model
        
        # generate ordereddict to access all submodules
        self.structure = recursive_module_dict(self.model)
        
        # initialize hooks
        self.init_hooks()

    def init_hooks(self):
        """Creates basic hooks for every layer in a model which just keep track of the output."""

        self.hooks = OrderedDict(basic_hooks=OrderedDict(), custom_hooks=OrderedDict())

        # recursive hooking function
        def hook_layers(net, prefix=[]):
            if hasattr(net, "_modules"):
                for name, layer in net._modules.items():
                    if layer is None:
                        # e.g. GoogLeNet's aux1 and aux2 layers
                        continue
                    self.hooks["basic_hooks"]["->".join(prefix + [name])] = ModuleHook(layer)
                    hook_layers(layer, prefix=prefix + [name])
            else:
                raise ValueError('net has not _modules attribute! Check if your model is properly instantiated..')

        hook_layers(self.model)
    
    def __call__(self, name):
        if name in self.hooks["basic_hooks"]:
            category = "basic_hooks"
        elif name in self.hooks["custom_hooks"]:
            category = "custom_hooks"
        else:
            raise ValueError(f"Unknown layer {name}. Retrieve the list of layers with instance method `.get_names()`")


        if self.hooks[category][name] is None:
            raise ValueError("There are no saved feature maps. Make sure to put the model in eval mode, like so: `model.to(device).eval()`.")
        
        return self.hooks[category][name].features
    
    def __getitem__(self, name):
        return self(name)

    def __repr__(self):
        repr_str = \
              "\nBasic Hooks:\n" \
            + "------------\n" \
            + "\n".join(self.hooks["basic_hooks"].keys()) \
            + "\n\n" \
            + "Custom Hooks:\n" \
            + "-------------\n" \
            + "\n".join(self.hooks["custom_hooks"].keys()) \
            + "\n"
            
        return repr_str

    def items(self):
        return chain(self.hooks["basic_hooks"].items(), self.hooks["custom_hooks"].items())
    
    def get_names(self):
        return list(chain(self.hooks["basic_hooks"].keys(), self.hooks["custom_hooks"].keys()))
    
    def add_custom_hook(
        self, 
        layer_name: str, 
        hook_key: str, 
        output_fn: Callable,
    ) -> None: # currently only forward hooks
        
        # check if hook name already exists
        if hook_key in self.hooks["basic_hooks"]:
            raise ValueError(f"Hook with name {hook_key} already exist in self.basic_hooks!")
        if hook_key in self.hooks["custom_hooks"]:
            raise ValueError(f"Hook with name {hook_key} already exist in self.custom_hooks!")

        # parse name
        keys = layer_name.split('->')
        
        # traverse structure and get module
        if len(keys) > 0:
            module = self.structure['children'][keys[0]]
            for k in keys[1:]:
                module = module['children'][k]
        module = module['module']

        self.hooks["custom_hooks"][hook_key] = ModuleHook(module, output_fn)
    
    def remove_hook(self, hook_key):
        # check if hook name already exists and delete it if so
        if hook_key in self.hooks["basic_hooks"]:
            self.hooks["basic_hooks"][hook_key].close()
            del self.hooks["basic_hooks"][hook_key]
        if hook_key in self.hooks["custom_hooks"]:
            self.hooks["custom_hooks"][hook_key].close()
            del self.hooks["custom_hooks"][hook_key]
        else:
            logging.warn(f'Could not find hook {hook_key} to delete!')


def recursive_module_dict(model: torch.nn.Module) -> OrderedDict:
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
    slice_ = create_slice(indices, replacement_tensor.shape)
    def func(output):
        # add dummy dimensions if shape mismatch
        diff = len(output[slice_].shape) - len(replacement_tensor.shape)
        rep = replacement_tensor
        for i in range(diff):
            rep = rep[None]
        # replace part of tensor    
        output[slice_] = rep.to(output.device)
        return output

    return func

def create_slice(indices, target_shape=None):
    if target_shape is not None:
        trailing = len(target_shape) * ",:"
    else:
        trailing = ""
    slice_ = eval(f'np.s_[...,{indices}{trailing}]')
    return slice_


# special to ROME reimplementation
def hidden_patch_hook_fn(position, replacement_tensor):
    inner = replace_activation(str(position), replacement_tensor)    
    def func(output):
        output[0][...] = inner(output[0])
        return output
    return func

def mlp_patch_interval(center, width, replacement_tensor):
    inner = replace_activation(f"{center-width//2}:{center+width//2}", replacement_tensor)
    def func(output):
        output[...] = innter(output)
        return output
    return func

def attn_patch_interval(center, width, replacement_tensor):
    pass #TODO
 
def additive_noise(indices, mean=0, std=0.1):
    slice_ = create_slice(indices)
    def func(output):
        noise = mean + std * torch.randn_like(output[slice_])
        output[slice_] += noise
        return output
    return func