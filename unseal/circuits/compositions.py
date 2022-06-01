import itertools
from typing import Optional, Tuple

import einops
import torch

from . import Attention, Tensor
from .utils import get_o_weight, get_qkv_weights


def composition(a: Tensor, b: Tensor, mode: Optional[str] = 'fro') -> float:
    if mode == 'fro':
        return (a @ b).norm(p='fro') / (a.norm(p='fro') * b.norm(p='fro'))
    else:
        raise NotImplementedError(f"composition mode {mode} is not implemented yet!")

def q_composition(qk: Tensor, ov: Tensor) -> float:
    return composition(qk.T, ov)

def k_composition(qk: Tensor, ov: Tensor) -> float:
    return composition(qk, ov)

def v_composition(ov_2: Tensor, ov_1: Tensor):
    return composition(ov_2, ov_1)


@torch.no_grad()
def compute_all_compositions(attn_1: Attention, attn_2: Attention, baselines: Optional[Tuple[float, float, float]] = None):
    
    q_1, k_1, v_1 = get_qkv_weights(attn_1)
    o_1 = get_o_weight(attn_1)
    ov_1 = torch.einsum('abc, acd -> abd', einops.rearrange(o_1, 'a c b -> a b c'), v_1)
    qk_1 = torch.einsum('abc, acd -> abd', einops.rearrange(q_1, 'a c b -> a b c'), k_1)
    
    q_2, k_2, v_2 = get_qkv_weights(attn_2)
    o_2 = get_o_weight(attn_2)
    ov_2 = torch.einsum('abc, acd -> abd', einops.rearrange(o_2, 'a c b -> a b c'), v_2)
    qk_2 = torch.einsum('abc, acd -> abd', einops.rearrange(q_2, 'a c b -> a b c'), k_2)
    
    q_comps = []
    k_comps = []
    v_comps = []
    
    for head_1, head_2 in itertools.product(range(attn_1.num_heads), range(attn_2.num_heads)):
        q_comps.append(q_composition(qk_2[head_2], ov_1[head_1]))
        k_comps.append(k_composition(qk_2[head_2], ov_1[head_1]))
        v_comps.append(v_composition(ov_2[head_2], ov_1[head_1]))
    
    if baselines is not None:
        q_comps = torch.stack(q_comps) - baselines[0]
        k_comps = torch.stack(k_comps) - baselines[1]
        v_comps = torch.stack(v_comps) - baselines[2]
    else:    
        q_comps = torch.stack(q_comps)
        k_comps = torch.stack(k_comps)
        v_comps = torch.stack(v_comps)
    
    return q_comps.clamp(min=0).numpy(), k_comps.clamp(min=0).numpy(), v_comps.clamp(min=0).numpy()
