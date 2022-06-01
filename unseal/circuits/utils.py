import math
from typing import Tuple, Optional

import einops
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention

from . import Attention, Tensor

# TODO: both getter functions don't yet work if you have biases in the qkv and o modules.


def get_qkv_weights(attention_module: Attention) -> Tuple[Tensor, Tensor, Tensor]:
    """Extracts the q, k, and v weights from an attention module. Matrices are reshaped to (num_heads, head_dim, out_dim)

    :param attention_module: Attention module.
    :type attention_module: Attention
    :raises ModuleNotFoundError: Unknown attention module type.
    :return: Tuple of q, k, and v weight matrices of the attention module.
    :rtype: Tuple[Tensor, Tensor, Tensor]
    """
    if isinstance(attention_module, GPT2Attention):
        q, k, v = attention_module.c_proj.weight.chunk(3, dim=0)
    else:
        try:
            q, k, v = attention_module.q_proj.weight, attention_module.k_proj.weight, attention_module.v_proj.weight
        except:
            raise ModuleNotFoundError(f"{attention_module} does not have q_proj, k_proj, v_proj")
        
    q = einops.rearrange(q, '(num_heads head_dim) out_dim -> num_heads head_dim out_dim', head_dim=attention_module.head_dim)
    k = einops.rearrange(k, '(num_heads head_dim) out_dim -> num_heads head_dim out_dim', head_dim=attention_module.head_dim)
    v = einops.rearrange(v, '(num_heads head_dim) out_dim -> num_heads head_dim out_dim', head_dim=attention_module.head_dim)
    
    return q, k, v

def get_o_weight(attention_module: Attention) -> Tensor:
    """Extracts the output projection matrix from an attention module. Matrix is reshaped to (num_heads, head_dim, out_dim)

    :param attention_module: Attention module.
    :type attention_module: Attention
    :raises TypeError: If unknown attention module type.
    :return: Output weight matrix of the attention module.
    :rtype: Tensor
    """
    if not isinstance(attention_module, Attention):
        raise TypeError(f"{attention_module} is not an instance of Attention, has type {type(attention_module)}")
    
    if isinstance(attention_module, GPT2Attention):
        return einops.rearrange(attention_module.c_proj.weight, 'out_dim (num_heads head_dim) -> num_heads head_dim out_dim', head_dim=attention_module.head_dim)
    else:
        return einops.rearrange(attention_module.out_proj.weight, 'out_dim (num_heads head_dim) -> num_heads head_dim out_dim', head_dim=attention_module.head_dim)

def kaiming_uniform_limits(
    weight: Tensor, 
    gain: Optional[float] = 1.0, 
    mode: Optional[str] = 'fan_in'
) -> Tuple[float, float]:
    """Returns the xavier uniform limits for a given weight matrix.

    :param weight: Weight matrix of shape (fan_out, fan_in)
    :type weight: Tensor
    :param gain: Gain used to initialize the weight matrix, defaults to 1.0
    :type gain: Optional[float], optional
    :return: Tuple of lower and upper limits.
    :rtype: Tuple[float, float]
    """
    if mode == 'fan_in':
        return (-gain * math.sqrt(3 / (weight.shape[1])), gain * math.sqrt(3 / (weight.shape[1])))
    elif mode == 'fan_out':
        return (-gain * math.sqrt(3 / (weight.shape[0])), gain * math.sqrt(3 / (weight.shape[0])))
    else:
        raise ValueError(f"mode {mode} not supported")
 
def xavier_uniform_limits(weight: Tensor, gain: Optional[float] = 1.0) -> Tuple[float, float]:
    """Returns the xavier uniform limits for a given weight matrix.

    :param weight: Weight matrix.
    :type weight: Tensor
    :param gain: Gain used to initialize the weight matrix, defaults to 1.0
    :type gain: Optional[float], optional
    :return: Tuple of lower and upper limits.
    :rtype: Tuple[float, float]
    """
    return (-gain * math.sqrt(6 / (weight.shape[0] + weight.shape[1])), gain * math.sqrt(6 / (weight.shape[0] + weight.shape[1])))

def uniform_limits(
    weight: Tensor,
    dist: Optional[str] = 'xavier',
    dist_kwargs: Optional[dict] = None,
) -> Tuple[float, float]:
    """Get the limits for a given weight matrix.

    :param weight: Weight matrix of shape (fan_out, fan_in)
    :type weight: Tensor
    :param dist: Distribution to sample from, one of ['xavier', 'kaiming'], defaults to 'xavier'
    :type dist: Optional[str], optional
    :param dist_kwargs: Kwargs to pass on to the distribution function, defaults to None
    :type dist_kwargs: Optional[dict], optional
    :raises ValueError: Unknown distribution.
    :return: Limits of the uniform distribution.
    :rtype: Tuple[float, float]
    """
    if dist == 'xavier':
        return xavier_uniform_limits(weight, **(dist_kwargs or {}))
    elif dist == 'kaiming':
        return kaiming_uniform_limits(weight, **(dist_kwargs or {}))
    else:
        raise ValueError(f"distribution {dist} not supported")


    
    