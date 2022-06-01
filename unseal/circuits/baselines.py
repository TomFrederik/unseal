from typing import Optional, Tuple

import torch

from . import Attention, Tensor
from .compositions import k_composition, q_composition, v_composition
from .utils import get_o_weight, get_qkv_weights, uniform_limits


def sample_single_matrix_uniform(
    low: float, 
    high: float, 
    shape: torch.Size,
) -> Tensor:
    """Samples a matrix from a uniform distribution with given limits.

    :param low: Lower bound of the uniform distribution.
    :type low: float
    :param high: Upper bound of the uniform distribution.
    :type high: float
    :param shape: Shape of the matrix.
    :type shape: torch.Size
    :return: Sampled matrix.
    :rtype: Tensor
    """
    return torch.distributions.Uniform(low, high).rsample(shape)

def sample_composed_matrix_uniform(
    shape_1: torch.Size, 
    limits_1: Tuple[float, float], 
    shape_2: torch.Size, 
    limits_2: Tuple[float, float],
) -> Tensor:
    """Samples two matrices A and B from uniform distributions with given limits and return the product A^T @ B

    :param shape_1: Shape of matrix A.
    :type shape_1: torch.Size
    :param limits_1: Limits of matrix A.
    :type limits_1: Tuple[float, float]
    :param shape_2: Shape of matrix B.
    :type shape_2: torch.Size
    :param limits_2: Limits of matrix B.
    :type limits_2: Tuple[float, float]
    :raises ValueError: Shape mismatch.
    :return: Composition of the sampled matrices, A^T @ B
    :rtype: Tensor
    """
    if shape_1[0] != shape_2[0]:
        raise ValueError('Shapes must have the same first dimension.')
    
    mat_1 = sample_single_matrix_uniform(limits_1[0], limits_1[1], shape_1)
    mat_2 = sample_single_matrix_uniform(limits_2[0], limits_2[1], shape_2)
    return mat_1.T @ mat_2    

def sample_qk_matrix_uniform(
    shape: torch.Size, 
    limits: Tuple[float, float],
) -> Tensor:
    """Sample QK matrix with given shapes from a uniform distribution with given limits.

    :param shape: Matrix shape
    :type shape: torch.Size
    :param limits: Limits of the uniform distribution.
    :type limits: Tuple[float, float]
    :return: Sampled QK matrix.
    :rtype: Tensor
    """
    return sample_composed_matrix_uniform(shape, limits, shape, limits)

def sample_ov_matrix_uniform(
    shape: torch.Size, 
    o_limits: Tuple[float, float], 
    v_limits: Tuple[float, float],
) -> Tensor:
    """Sample OV matrix with given shapes from a uniform distribution with given limits.

    :param shape: Shape of O and V matrices.
    :type shape: torch.Size
    :param o_limits: Limits of the uniform distribution for O.
    :type o_limits: Tuple[float, float]
    :param v_limits: Limits of the uniform distribution for V.
    :type v_limits: Tuple[float, float]
    :return: Sampled OV matrix.
    :rtype: Tensor
    """
    return sample_composed_matrix_uniform(shape, o_limits, shape, v_limits)

def compute_v_comp_baseline(
    shape: torch.Size, 
    qkv_limits: Tuple[float, float], 
    o_limits: Tuple[float, float], 
    num_samples: int,
) -> float:
    """Computes the baseline for V composition assuming uniform distribution.

    :param shape: Shape of the weight matrices.
    :type shape: torch.Size
    :param qkv_limits: Limits of the uniform distribution for QKV.
    :type qkv_limits: Tuple[float, float]
    :param o_limits: Limits of the uniform distribution for O.
    :type o_limits: Tuple[float, float]
    :param num_samples: Number of samples used to approximate the baseline.
    :type num_samples: int
    :return: Baseline V composition.
    :rtype: float
    """
    cur = 0
    for i in range(num_samples):
        mat_1 = sample_ov_matrix_uniform(shape, o_limits, qkv_limits)
        mat_2 = sample_ov_matrix_uniform(shape, o_limits, qkv_limits)
        cur += v_composition(mat_1, mat_2)
    return cur.item() / num_samples

def compute_q_comp_baseline(
    shape: torch.Size, 
    qkv_limits: Tuple[float, float], 
    o_limits: Tuple[float, float], 
    num_samples: int,
) -> float:
    """Computes the baseline for Q composition assuming uniform distribution.

    :param shape: Shape of the weight matrices.
    :type shape: torch.Size
    :param qkv_limits: Limits of the uniform distribution for QKV.
    :type qkv_limits: Tuple[float, float]
    :param o_limits: Limits of the uniform distribution for O.
    :type o_limits: Tuple[float, float]
    :param num_samples: Number of samples used to approximate the baseline.
    :type num_samples: int
    :return: Baseline Q composition.
    :rtype: float
    """
    cur = 0
    for i in range(num_samples):
        qk_mat = sample_qk_matrix_uniform(shape, qkv_limits)
        ov_mat = sample_ov_matrix_uniform(shape, o_limits, qkv_limits)
        cur += q_composition(qk_mat, ov_mat)
    return cur.item() / num_samples

def compute_k_comp_baseline(shape: torch.Size, 
    qkv_limits: Tuple[float, float], 
    o_limits: Tuple[float, float], 
    num_samples: int,
) -> float:
    """Computes the baseline for K composition assuming uniform distribution.

    :param shape: Shape of the weight matrices.
    :type shape: torch.Size
    :param qkv_limits: Limits of the uniform distribution for QKV.
    :type qkv_limits: Tuple[float, float]
    :param o_limits: Limits of the uniform distribution for O.
    :type o_limits: Tuple[float, float]
    :param num_samples: Number of samples used to approximate the baseline.
    :type num_samples: int
    :return: Baseline K composition.
    :rtype: float
    """
    cur = 0
    for i in range(num_samples):
        qk_mat = sample_qk_matrix_uniform(shape, qkv_limits)
        ov_mat = sample_ov_matrix_uniform(shape, o_limits, qkv_limits)
        cur += k_composition(qk_mat, ov_mat)
    return cur.item() / num_samples

def compute_all_baselines(
    attention_module: Attention, 
    num_samples: int,
    dist: Optional[str] = 'xavier',
    dist_kwargs: Optional[dict] = None,
) -> Tuple[float, float, float]:
    """Compute Q, K, and V composition score baselines for the given attention module.

    :param attention_module: Attention module
    :type attention_module: Attention
    :param num_samples: Number of samples used to approximate the baseline.
    :type num_samples: int
    :return: Q, K and V composition score baselines.
    :rtype: Tuple[float, float, float]
    """
    q, k, v = get_qkv_weights(attention_module)
    o = get_o_weight(attention_module)
    
    qkv_shape = q.shape[1:]
    
    
    qkv_limits = uniform_limits(torch.cat([q, k, v], dim=1), dist=dist, dist_kwargs=dist_kwargs)
    o_limits = uniform_limits(o, dist=dist, dist_kwargs=dist_kwargs)
    
    q_comp_baseline = compute_q_comp_baseline(qkv_shape, qkv_limits, o_limits, num_samples)
    k_comp_baseline = compute_k_comp_baseline(qkv_shape, qkv_limits, o_limits, num_samples)
    v_comp_baseline = compute_v_comp_baseline(qkv_shape, qkv_limits, o_limits, num_samples)
    
    return q_comp_baseline, k_comp_baseline, v_comp_baseline
