import logging
from typing import Optional, List

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from .transformers_util import get_num_layers
from .hooks.common_hooks import create_logit_hook
from .hooks.commons import HookedModel

def generate_logit_lense(
    model: HookedModel, 
    tokenizer: AutoTokenizer, 
    sentence: str,
    layers: Optional[List[int]] = None,
    ranks: Optional[bool] = False,
    kl_div: Optional[bool] = False,
    include_input: Optional[bool] = False,
    layer_key_prefix: Optional[str] = None,
):
    """Generates the necessary data to generate the plots from the logits `lense post 
    <https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens>`_.

    Returns None for ranks and kl_div if not specified.

    :param model: Model that is investigated.
    :type model: HookedModel
    :param tokenizer: Tokenizer of the model.
    :type tokenizer: AutoTokenizer
    :param sentence: Sentence to be analyzed.
    :type sentence: str
    :param layers: List of layers to be investigated.
    :type layers: Optional[List[int]]
    :param ranks: Whether to return ranks of the correct token throughout layers, defaults to False
    :type ranks: Optional[bool], optional
    :param kl_div: Whether to return the KL divergence between intermediate probabilities and final output probabilities, defaults to False
    :type kl_div: Optional[bool], optional
    :param include_input: Whether to include the immediate logits/ranks/kld after embedding the input, defaults to False
    :type include_input: Optional[bool], optional
    :param layer_key_prefix: Prefix for the layer keys, e.g. 'transformer->h' for GPT like models, defaults to None
    :type layer_key_prefix: Optional[str], optional
    :return: logits, ranks, kl_div
    :rtype: Tuple[torch.Tensor]
    """
    
    # TODO
    if include_input:
        logging.warning("include_input is not implemented yet")
    
    # prepare model input
    tokenized_sentence = tokenizer.encode(sentence, return_tensors='pt').to(model.device)
    targets = tokenizer.encode(sentence)[1:]
    
    # instantiate hooks
    num_layers = get_num_layers(model, layer_key_prefix=layer_key_prefix)
    if layers is None:
        layers = list(range(num_layers))
    logit_hooks = [create_logit_hook(layer, model, 'lm_head', layer_key_prefix) for layer in layers]
    
    # run model
    model.forward(tokenized_sentence, hooks=logit_hooks)
    logits = torch.stack([model.save_ctx[str(layer) + '_logits']['logits'] for layer in range(num_layers)], dim=0)
    logits = F.log_softmax(logits, dim=-1)
    
    # compute ranks and kld
    if ranks:
        inverted_ranks = torch.argsort(logits, dim=-1, descending=True)
        ranks = torch.argsort(inverted_ranks, dim=-1) + 1
        ranks = ranks[:, torch.arange(len(targets)), targets]
    else:
        ranks = None

    if kl_div: # Note: logits are already normalized internally by the logit_hook
        kl_div = F.kl_div(logits, logits[-1][None], reduction='none', log_target=True).sum(dim=-1)
        kl_div = kl_div[:, torch.arange(len(targets)), targets]
    else:
        kl_div = None    
        
    logits = logits[:, torch.arange(len(targets)), targets]
    
    return logits, ranks, kl_div


