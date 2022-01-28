from collections import OrderedDict
from typing import Callable, Optional
from functools import partial

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


def recursive_module_dict(model: torch.nn.Module) -> OrderedDict:
    """Recursively generates an OrderedDict representing the module structure in a nn.Module.

    :param model: The (sub-)module for which to generate the structure
    :type model: torch.nn.Module
    :return: Structure of the module
    :rtype: OrderedDict
    """
    if len(model._modules) == 0:
        return model
    
    subdict = OrderedDict(module=model, children=OrderedDict())
    for name, submodule in model._modules.items():
        subdict['children'][name] = recursive_module_dict(submodule)
    
    return subdict


def hook_model(model: torch.nn.Module) -> Callable:
    """Creates hooks for every layer in a model.

    :param model: the model to be hooked
    :type model: torch.nn.Module
    :raises TypeError: model is wrong type
    :return: ``hook`` that returns the models features at a given layer (for the last-processed input)
    :rtype: Callable
    """
    
    # check inputs
    if not isinstance(model, torch.nn.Module):
        raise TypeError(f"model should be type torch.nn.Module but is {type(model)}")

    # generate ordereddict to access all submodules
    structure = recursive_module_dict(model)

    hooks = OrderedDict()
    # recursive hooking function
    def hook_layers(net, prefix=[]):
        if hasattr(net, "_modules"):
            for name, layer in net._modules.items():
                if layer is None:
                    # e.g. GoogLeNet's aux1 and aux2 layers
                    continue
                hooks["->".join(prefix + [name])] = ModuleHook(layer)
                hook_layers(layer, prefix=prefix + [name])
        else:
            raise ValueError('net has not _modules attribute! Check if your model is properly instantiated..')

    hook_layers(model)
    
    # helper function for error handling
    def helper(name):
        if not name in hooks:
            raise ValueError(f"Unknown layer {name}. Retrieve the list of layers with `model_utils.get_model_layers(model)`")
        if hooks[name] is None:
            raise ValueError("There are no saved feature maps. Make sure to put the model in eval mode, like so: `model.to(device).eval()`.")
        return hooks[name].features
    
    return helper


def hidden_patch_hook(base_hooks, layer, position, replacement_tensor):
    
    def func(output):
        output[0][position] = replacement_tensor
        return output

    base_hooks[layer] = ModuleHook(layer, output_fn=func)
    return base_hooks


class FullModelHooks:
    def __init__(self, model):
        # check inputs
        if not isinstance(model, torch.nn.Module):
            raise TypeError(f"model should be type torch.nn.Module but is {type(model)}")
        
        self.model = model
        
        # generate ordereddict to access all submodules
        self.structure = recursive_module_dict(self.model)
        
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

    def get_names(self):
        return list(self.hooks["basic_hooks"].keys()) + list(self.hooks["custom_hooks"].keys())
    
    def add_custom_hook(
        self, 
        layer_name: str, 
        hook_key: str, 
        output_fn: Callable,
    ) -> None: # currently only forward hooks
        
        # check if hook name already exists
        if hook_key in self.hooks["basic_hooks"]:
            raise ValueError(f"Hook with name {hook_name} already exist in self.basic_hooks!")
        if hook_key in self.hooks["custom_hooks"]:
            raise ValueError(f"Hook with name {hook_name} already exist in self.custom_hooks!")

        # parse name
        keys = layer_name.split('->')
        
        # traverse structure and get module
        if len(keys) > 0:
            module = self.structure['children'][keys[0]]
            for k in keys[1:]:
                module = module['children'][k]
        module = module['module']

        self.hooks["custom_hooks"][hook_key] = ModuleHook(module, output_fn)
