from collections import OrderedDict
from typing import Callable

import pytest
import torch
from unseal.hooks import commons

class TestHookedModel():
    def test_constructor(self):
        model = commons.HookedModel(torch.nn.Module())
        assert model is not None
        assert isinstance(model.model, torch.nn.Module)
        assert isinstance(model.save_ctx, dict)
        assert isinstance(model.layers, OrderedDict)
        
        
        with pytest.raises(TypeError):
            model = commons.HookedModel('not a module')
    
    def test_init_refs(self):
        model = commons.HookedModel(
            torch.nn.Sequential(
                torch.nn.Sequential(
                    torch.nn.Linear(10,10)
                ),
                torch.nn.Linear(10,10)
            )
        )
        assert list(model.layers.keys()) == ['0', '0->0', '1']
    
    def test__hook_wrapper(self):
        model = commons.HookedModel(torch.nn.Module())
        def test_func(save_ctx, inp, output):
            return 1
        
        func = model._hook_wrapper(test_func, 'key')
        assert isinstance(func, Callable)
        
    def test_get_ctx_keys(self):
        model = commons.HookedModel(torch.nn.Module())
        assert model.get_ctx_keys() == []
        
        model.save_ctx['key'] = dict()
        assert model.get_ctx_keys() == ['key']
    
    def test_repr(self):
        model = commons.HookedModel(torch.nn.Module())
        assert model.__repr__() == model.model.__repr__()
        
    def test_device(self):
        model = commons.HookedModel(torch.nn.Linear(10, 10))
        assert model.device.type == 'cpu'
        
        if torch.cuda.is_available():
            model.to('cuda')
        assert model.device.type == 'cuda'
    
    def test_forward(self):
        model = commons.HookedModel(torch.nn.Sequential(torch.nn.Linear(10, 10)))
        def fn(save_ctx, input, output):
            save_ctx['key'] = 1
        hook = commons.Hook('0', fn, 'key')
        model.forward(torch.rand(10), [hook])
        assert model.save_ctx['key']['key'] == 1
    
    def test___call__(self):
        model = commons.HookedModel(torch.nn.Sequential(torch.nn.Linear(10, 10)))
        def fn(save_ctx, input, output):
            save_ctx['key'] = 2
        hook = commons.Hook('0', fn, 'key')
        model(torch.rand(10), [hook])
        assert model.save_ctx['key']['key'] == 2


def test_hook():
    def correct_func(save_ctx, input, output):
        save_ctx['key'] = 1
    hook = commons.Hook('0', correct_func, 'key')
    assert hook.layer_name == '0'
    assert hook.func == correct_func
    assert hook.key == 'key'
    
    def false_func(save_ctx):
        save_ctx['key'] = 1
    with pytest.raises(TypeError):
        hook = commons.Hook('0', false_func, 'key')