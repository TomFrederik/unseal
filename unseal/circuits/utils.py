import itertools
import math
from typing import Tuple, TypeVar

import einops
import torch
from torch import Tensor
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
from transformers.models.gpt_neo.modeling_gpt_neo import GPTNeoSelfAttention
from transformers.models.gptj.modeling_gptj import GPTJAttention

Attention = TypeVar('Attention', GPT2Attention, GPTNeoSelfAttention, GPTJAttention)

def get_qkv_weights(attention_module: Attention) -> Tuple[Tensor, Tensor, Tensor]:
    if isinstance(attention_module, GPT2Attention):
        q, k, v = attention_module.c_proj.weight.chunk(3)
        print(f"{q.shape = }")
        print(f"{k.shape = }")
        print(f"{v.shape = }")
    else:
        q, k, v = attention_module.q_proj.weight, attention_module.k_proj.weight, attention_module.v_proj.weight
    
    q = einops.rearrange(q, '(num_heads head_dim) out_dim -> num_heads head_dim out_dim', head_dim=attention_module.head_dim)
    k = einops.rearrange(k, '(num_heads head_dim) out_dim -> num_heads head_dim out_dim', head_dim=attention_module.head_dim)
    v = einops.rearrange(v, '(num_heads head_dim) out_dim -> num_heads head_dim out_dim', head_dim=attention_module.head_dim)
    
    return q, k, v

def get_o_weight(attention_module: Attention) -> Tensor:
    if not isinstance(attention_module, Attention):
        raise TypeError(f"{attention_module} is not an instance of Attention, has type {type(attention_module)}")
    
    if isinstance(attention_module, GPT2Attention):
        return einops.rearrange(attention_module.c_proj.weight, '(num_heads head_dim) out_dim -> num_heads head_dim out_dim', head_dim=attention_module.head_dim)
    else:
        return einops.rearrange(attention_module.out_proj.weight, '(num_heads head_dim) out_dim -> num_heads head_dim out_dim', head_dim=attention_module.head_dim)

def composition(a: Tensor, b: Tensor):
    return (a.T @ b).norm(p='fro') / (a.norm(p='fro') * b.norm(p='fro'))

def q_composition(qk: Tensor, ov: Tensor):
    return composition(qk, ov)

def k_composition(qk: Tensor, ov: Tensor):
    return composition(qk.T, ov)

def v_composition(ov_2: Tensor, ov_1: Tensor):
    return composition(ov_2.T, ov_1)

def get_init_limits(weight: Tensor) -> float:
    # compute xavier uniform initialization limits
    return (-math.sqrt(6/(weight.shape[0] + weight.shape[1])), math.sqrt(6/(weight.shape[0] + weight.shape[1])))

def approx_baseline(shape_1, shape_2, limits_1, limits_2, num_samples, device='cpu'):
    baseline = 0
    for i in range(num_samples):
        mat_1 = torch.distributions.Uniform(limits_1[0], limits_1[1]).rsample(shape_1).to(device)
        mat_2 = torch.distributions.Uniform(limits_2[0], limits_2[1]).rsample(shape_2).to(device)
        
        baseline += composition(mat_1, mat_2)
    
    return baseline / num_samples

def compute_all_baselines(attention_module: Attention, num_samples):
    device = next(attention_module.parameters()).device

    q, k, v = get_qkv_weights(attention_module)
    o = get_o_weight(attention_module)
    # print(f"q: {q.shape}")
    # print(f"k: {k.shape}")
    # print(f"v: {v.shape}")
    # print(f"o: {o.shape}")
    qk_shape = (q.shape[1],) + (k.shape[1],)
    ov_shape = (o.shape[1],) + (v.shape[1],)
    # print(f"{qk_shape = }")
    # print(f"{ov_shape = }")
    qk_limits = get_init_limits(attention_module.qkv_proj.weight)
    ov_limits = get_init_limits(o)
    
    qk_baseline = approx_baseline(qk_shape, ov_shape, qk_limits, ov_limits, num_samples, device)
    v_baseline = approx_baseline(ov_shape, ov_shape, ov_limits, ov_limits, num_samples, device)
        
    return qk_baseline, v_baseline

@torch.no_grad()
def compute_all_compositions(attn_1: Attention, attn_2: Attention, num_samples: int = 1000, subtract_baseline: bool = False):
    if subtract_baseline:
        qk_baseline, v_baseline = compute_all_baselines(attn_2, num_samples)
    else:
        qk_baseline, v_baseline = 0, 0
    # print(f"{qk_baseline = }")
    # print(f"{v_baseline = }")
    
    q_1, k_1, v_1 = get_qkv_weights(attn_1)
    # print(f"v_1: {v_1.shape}")
    o_1 = get_o_weight(attn_1)
    # print(f"o_1: {o_1.shape}")
    ov_1 = torch.einsum('abc, acd -> abd', einops.rearrange(o_1, 'a c b -> a b c'), v_1)
    qk_1 = torch.einsum('abc, acd -> abd', einops.rearrange(q_1, 'a c b -> a b c'), k_1)
    # print(f"{qk_1.shape = }")
    # print(f"{ov_1.shape = }")
    q_2, k_2, v_2 = get_qkv_weights(attn_2)
    o_2 = get_o_weight(attn_2)
    ov_2 = torch.einsum('abc, acd -> abd', einops.rearrange(o_2, 'a c b -> a b c'), v_2)
    qk_2 = torch.einsum('abc, acd -> abd', einops.rearrange(q_2, 'a c b -> a b c'), k_2)
    # print(f"{qk_2.shape = }")
    # print(f"{ov_2.shape = }")
    
    q_comps = []
    k_comps = []
    v_comps = []
    
    for head_1, head_2 in itertools.product(range(attn_1.num_heads), range(attn_2.num_heads)):
        q_comps.append(q_composition(qk_2[head_2], ov_1[head_1]))
        k_comps.append(k_composition(qk_2[head_2], ov_1[head_1]))
        v_comps.append(v_composition(ov_2[head_2], ov_1[head_1]))
    
    q_comps = torch.stack(q_comps) - qk_baseline
    k_comps = torch.stack(k_comps) - qk_baseline
    v_comps = torch.stack(v_comps) - v_baseline
    
    
    return q_comps.clamp(min=0).cpu(), k_comps.clamp(min=0).cpu(), v_comps.clamp(min=0).cpu()
    
    