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

test_input = torch.rand((1,10))

replacement = torch.ones(1,1) + 100
test_fn = hidden_patch_hook_fn(1, replacement)
hooks.add_custom_hook('0', 'replacement', test_fn)

print(hooks)

out = model(test_input)
print(hooks['0'])
print(hooks('replacement'))
print(out)