from hooks import *

import torch.nn as nn


model = nn.Sequential(
    nn.Linear(10,5),
    nn.GELU(),
    nn.Sequential(
        nn.GELU(),
        nn.Linear(5,2),
    )
)

hooks = FullModelHooks(model)
hooks.add_custom_hook('2', 'set_0', lambda x: torch.zeros_like(x))
hooks.add_custom_hook('2', 'mul_2', lambda x: x*2)
hooks.add_custom_hook('2', 'add_1', lambda x: x+1)
print(hooks)
print(hooks.get_names())
test_input = torch.rand((1,10)) + 2

out = model(test_input)
print(hooks('2'))
print(hooks['set_0'])
print(hooks['add_1'])
print(hooks('mul_2'))
print(out)