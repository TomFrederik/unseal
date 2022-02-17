import logging
from typing import Optional

import torch
from transformers import AutoTokenizer

from .transformers_util import load_from_pretrained, get_num_layers
from .hooks.common_hooks import logit_hook
from .hooks.commons import HookedModel

def generate_logit_lense(
    model: HookedModel, 
    tokenizer: AutoTokenizer, 
    sentence: str,
    ranks: Optional[bool] = False,
    kl_div: Optional[bool] =False,
    include_input: Optional[bool] =False,
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
    :param ranks: Whether to return ranks of the correct token throughout layers, defaults to False
    :type ranks: Optional[bool], optional
    :param kl_div: Whether to return the KL divergence between intermediate probabilities and final output probabilities, defaults to False
    :type kl_div: Optional[bool], optional
    :param include_input: Whether to include the immediate logits/ranks/kld after embedding the input, defaults to False
    :type include_input: Optional[bool], optional
    :return: logits, ranks, kl_div
    :rtype: Tuple[torch.Tensor]
    """
    
    # TODO
    if include_input:
        logging.warning("include_input is not implemented yet")
    
    # prepare model input
    tokenized_sentence = tokenizer.encode(sentence, return_tensors='pt').to('cpu')
    targets = tokenizer.encode(sentence)[1:]
    
    # instantiate hooks
    num_layers = get_num_layers(model)
    logit_hooks = [logit_hook(layer, model) for layer in range(num_layers)]
    
    # run model
    model.forward(tokenized_sentence, hooks=logit_hooks)
    logits = torch.stack([model.save_ctx[str(layer) + '_logits']['logits'][0] for layer in range(num_layers)], dim=0)

    # compute ranks and kld
    # TODO, ranks seem off, not sure what's going on here.
    if ranks:
        raise NotImplementedError
        # inverted_ranks = torch.argsort(logits, dim=-1, descending=True)
        # print(f"{inverted_ranks = }")
        # ranks = torch.argsort(inverted_ranks, dim=-1) + 1
        # print(f"{ranks = }")
        # ranks = ranks[:, torch.arange(1, len(targets)+1), targets]
        # print(f"{ranks[:,13] = }")
        # print(f"{ranks.shape = }")
    else:
        ranks = None

    if kl_div:
        # compute kl_div between intermediate and final layers
        log_probs = torch.nn.LogSoftmax(dim=-1)(logits)
        kl_div_loss = torch.nn.KLDivLoss(reduction='none', log_target=True)
        kl_div = kl_div_loss(log_probs, log_probs[-1][None]).sum(dim=-1)
    else:
        kl_div = None    
    
    return logits, ranks, kl_div

# model, tokenizer, config = load_from_pretrained('gpt2-xl')
# model = hooks.HookedModel(model).to('cpu')

# text = "After 1 comes 2. After 2 comes 3. After 3 comes 4."
# generate_logit_lense(model, tokenizer, text, ranks=True, kl_div=True)
