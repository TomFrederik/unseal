import json
import gc
from typing import Optional, Callable, Dict

import einops
import pysvelte as ps
import torch
from transformers import AutoTokenizer

from ..hooks import HookedModel
from ..hooks.common_hooks import create_attention_hook, gpt_attn_wrapper

def compute_attn_logits(
    model: HookedModel,
    model_name: str, 
    tokenizer: AutoTokenizer, 
    num_layers: int, 
    text: str, 
    html_storage: Dict, 
    save_path: Optional[str] = None,
    attn_name: Optional[str] = 'attn',
    output_idx: Optional[int] = 2,
    layer_key_prefix: Optional[str] = None,
    out_proj_name: Optional[str] = 'out_proj',
    attn_suffix: Optional[str] = None,
    unembedding_key: Optional[str] = 'lm_head',
):
    # parse inputs
    if save_path is None:
        save_path = f"{model_name}.json" 
    if layer_key_prefix is None:
        layer_key_prefix = ""
    else:
        layer_key_prefix += "->"
    if attn_suffix is None or attn_suffix == "":
        attn_suffix = ""
    
    # tokenize without tokenization artifact -> needed for visualization
    tokenized_text = tokenizer.tokenize(text)
    tokenized_text = list(map(tokenizer.convert_tokens_to_string, map(lambda x: [x], tokenized_text))) 
    
    # encode text
    model_input = tokenizer.encode(text, return_tensors='pt').to(model.device)
    target_ids = tokenizer.encode(text)[1:]
    
    # compute attention pattern
    attn_hooks = [create_attention_hook(i, f'attn_layer_{i}', output_idx, attn_name, layer_key_prefix) for i in range(num_layers)]
    model.forward(model_input, hooks=attn_hooks, output_attentions=True)
    
    # compute logits
    for layer in range(num_layers):

        # wrap the _attn function to create logit attribution
        model.save_ctx[f'logit_layer_{layer}'] = dict()
        old_fn = wrap_gpt_attn(model, layer, target_ids, unembedding_key, attn_name, attn_suffix, layer_key_prefix, out_proj_name)

        # forward pass
        model.forward(model_input, hooks=[])
    
        # parse attentions for this layer
        attention = model.save_ctx[f"attn_layer_{layer}"]['attn'][0]
        attention = einops.rearrange(attention, 'h n1 n2 -> n1 n2 h')

        # parse logits
        if model_input.shape[1] > 1: # otherwise we don't have any logit attribution
            logits = model.save_ctx[f'logit_layer_{layer}']['logits']
            pos_logits = logits['pos']
            neg_logits = logits['neg']
            pos_logits = pad_logits(pos_logits)
            neg_logits = pad_logits(neg_logits)
            
            pos_logits = einops.rearrange(pos_logits, 'h n1 n2 -> n1 n2 h')
            neg_logits = einops.rearrange(neg_logits, 'h n1 n2 -> n1 n2 h')
        else:
            pos_logits = torch.zeros((attention.shape[0], attention.shape[1], attention.shape[2]))
            neg_logits = torch.zeros((attention.shape[0], attention.shape[1], attention.shape[2]))
        
        # compute and display the html object
        html_object = ps.AttentionLogits(tokens=tokenized_text, attention=attention, pos_logits=pos_logits, neg_logits=neg_logits, head_labels=[f'{layer}:{j}' for j in range(attention.shape[-1])])
        html_object = html_object.update_meta(suppress_title=True)
        html_str = html_object.html_page_str()

        # save html string
        html_storage[f'layer_{layer}'] = html_str
        
        # reset _attn function
        reset_attn_fn(model, layer, old_fn, attn_name, attn_suffix, layer_key_prefix)
        
        # save progress so far
        with open(save_path, "w") as f: 
            json.dump(html_storage, f)

        # garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    return html_storage

def pad_logits(logits):
    logits = torch.cat([torch.zeros_like(logits[:,0][:,None]), logits], dim=1)
    logits = torch.cat([logits, torch.zeros_like(logits[:,:,0][:,:,None])], dim=2)
    return logits
    
def wrap_gpt_attn(
    model: HookedModel, 
    layer: int, 
    target_ids: Callable,
    unembedding_key: str,
    attn_name: Optional[str] = 'attn',
    attn_suffix: Optional[str] = None,
    layer_key_prefix: Optional[str] = None,
    out_proj_name: Optional[str] = 'out_proj',
) -> Callable:
    # parse inputs
    if layer_key_prefix is None:
        layer_key_prefix = ""
    if attn_suffix is None:
        attn_suffix = ""

    attn_name = f"{layer_key_prefix}{layer}->{attn_name}{attn_suffix}"
    out_proj_name = attn_name + f"->{out_proj_name}"
    
    model.layers[attn_name]._attn, old_fn = gpt_attn_wrapper(
        model.layers[attn_name]._attn,
        model.save_ctx[f'logit_layer_{layer}'], 
        model.layers[out_proj_name].weight,
        model.layers[unembedding_key].weight.T,
        target_ids
    )
    
    return old_fn
        

def reset_attn_fn(
    model: HookedModel, 
    layer: int, 
    old_fn: Callable,
    attn_name: Optional[str] = 'attn',
    attn_suffix: Optional[str] = None,
    layer_key_prefix: Optional[str] = None,
) -> None:
    # parse inputs
    if layer_key_prefix is None:
        layer_key_prefix = ""
    if attn_suffix is None:
        attn_suffix = ""
    
    # reset _attn function to old_fn
    del model.layers[f"{layer_key_prefix}{layer}->{attn_name}{attn_suffix}"]._attn
    model.layers[f"{layer_key_prefix}{layer}->{attn_name}{attn_suffix}"]._attn = old_fn