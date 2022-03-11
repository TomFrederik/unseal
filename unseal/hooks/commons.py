from collections import OrderedDict
from dataclasses import dataclass
from inspect import signature
from typing import List, Callable

import torch

from . import util

class Hook:
    layer_name: str
    func: Callable
    key: str
    def __init__(self, layer_name: str, func: Callable, key: str):
        # check that func takes three arguments
        if len(signature(func).parameters) != 3:
            raise TypeError(f'Hook function {func.__name__} should have three arguments, but has {len(signature(func).parameters)}.')

        self.layer_name = layer_name
        self.func = func
        self.key = key

class HookedModel(torch.nn.Module):
    def __init__(self, model):
        """Wrapper around a module that allows forward passes with hooks and a context object.

        :param model: Model to be hooked
        :type model: nn.Module
        :raises TypeError: Incorrect model type
        """
        super().__init__()

        # check inputs
        if not isinstance(model, torch.nn.Module):
            raise TypeError(f"model should be type torch.nn.Module but is {type(model)}")
        
        self.model = model
        
        # initialize hooks
        self.init_refs()

        # init context for accessing hook output
        self.save_ctx = dict()

    def init_refs(self):
        """Creates references for every layer in a model."""

        self.layers = OrderedDict()

        # recursive naming function
        def name_layers(net, prefix=[]):
            if hasattr(net, "_modules"):
                for name, layer in net._modules.items():
                    if layer is None:
                        # e.g. GoogLeNet's aux1 and aux2 layers
                        continue
                    self.layers["->".join(prefix + [name])] = layer
                    name_layers(layer, prefix=prefix + [name])
            else:
                raise ValueError('net has not _modules attribute! Check if your model is properly instantiated..')

        name_layers(self.model)


    def forward(
        self, 
        input_ids: torch.Tensor, 
        hooks: List[Hook],
        *args,
        **kwargs,
    ):
        """Wrapper around the default forward pass that temporarily registers hooks, executes the forward pass and then closes hooks again.
        """
        # register hooks
        registered_hooks = []
        for hook in hooks:
            layer = self.layers.get(hook.layer_name, None)
            if layer is None:
                raise ValueError(f'Layer {hook.layer_name} was not found during hook registration! Here is the whole model for reference:\n {next(iter(self.layers.items()))}')
            self.save_ctx[hook.key] = dict() # create sub-context for each hook to write to
            registered_hooks.append(layer.register_forward_hook(self._hook_wrapper(hook.func, hook.key)))

        # forward
        output = self.model(input_ids, *args, **kwargs) #TODO generalize to non-HF models which would not have an input_ids kwarg

        # remove hooks
        for hook in registered_hooks:
            hook.remove()
            
        return output

    def _hook_wrapper(self, func, hook_key):
        """Wrapper to comply with PyTorch's hooking API while enabling saving to context.

        :param func: [description]
        :type func: [type]
        :param hook_key: [description]
        :type hook_key: [type]
        :return: [description]
        :rtype: [type]
        """     
        return lambda model, input, output: func(save_ctx=self.save_ctx[hook_key], input=input[0], output=output)

    def get_ctx_keys(self):
        return list(self.save_ctx.keys())

    def __repr__(self):
        return self.model.__repr__()
    
    @property
    def device(self):
        return next(self.model.parameters()).device