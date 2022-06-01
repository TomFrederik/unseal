from typing import TypeVar

import torch
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
from transformers.models.gpt_neo.modeling_gpt_neo import GPTNeoSelfAttention
from transformers.models.gptj.modeling_gptj import GPTJAttention

Tensor = TypeVar('Tensor', bound=torch.Tensor)
Attention = TypeVar('Attention', GPT2Attention, GPTNeoSelfAttention, GPTJAttention)
