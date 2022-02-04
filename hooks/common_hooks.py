# pre-implemented common hooks
from typing import Iterable

import torch

from . import util

def save_output(save_ctx, input, output):
    if isinstance(output, torch.Tensor):
        save_ctx['output'] = output.to('cpu')
    elif isinstance(output, Iterable): # hope for the best
        save_ctx['output'] = util.recursive_to_device(output, 'cpu')