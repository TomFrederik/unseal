import json
import gc
from typing import List, Optional, Union, Iterable, Callable

import einops
import pysvelte as ps
import streamlit as st
import torch

from ..hooks import Hook
from ..hooks.common_hooks import gpt_get_attention_hook, gpt_attn_wrapper
from ..hooks import util 

def sample_text(col_idx, key):
    text = st.session_state[key]
    if st.session_state.prefix_prompt is not None and len(st.session_state.prefix_prompt) > 0:
        text = st.session_state.prefix_prompt + '\n' + text
    model_inputs = st.session_state.tokenizer.encode(text, return_tensors='pt').to(st.session_state.device)
    output = st.session_state.model.model.generate(model_inputs, **st.session_state.sample_kwargs, min_length=0, output_attentions=True)
    output_text = st.session_state.tokenizer.decode(output[0], skip_special_tokens=True)
    if st.session_state.prefix_prompt is not None and len(st.session_state.prefix_prompt) > 0:
        output_text = output_text.lstrip(st.session_state.prefix_prompt + '\n')
    
    st.session_state["storage"][col_idx] = output_text
    text_change(col_idx=col_idx)


def on_text_change(col_idx: Union[int, List[int]], text_key):
    if isinstance(col_idx, list):
        for idx in col_idx:
            on_text_change(idx, text_key)
    else:    
        st.session_state["storage"][col_idx] = st.session_state[text_key]
        text_change(col_idx)
    

## TODO
# move those two functions somewhere else
def grokking_get_attention(heads: Optional[Union[int, Iterable[int], str]] = None) -> Callable:
    # convert string to slice
    if heads is None:
        heads = ":"
    if isinstance(heads, str):
        heads = util.create_slice(heads)

    def func(save_ctx, input, output):
        save_ctx['attn'] = output[1][:,heads,...].detach().cpu()
    
    return func

def grokking_get_attention_hook(layer: int, key: str, heads: Optional[Union[int, Iterable[int], str]] = None) -> Callable:
    func = grokking_get_attention(heads)
    return Hook(f'transformer->{layer}->self_attn', func, key)    
####

def compute_attn_logits(text, save_destination):
    if st.session_state.model_name in st.session_state.registered_model_names:
        tokenized_text = st.session_state.tokenizer.tokenize(text)
        model_input = st.session_state.tokenizer.encode(text).to(st.session_state.device)
        target_ids = model_input[0,1:].to('cpu')
        #TODO generalize this somehow
        attn_hooks = [grokking_get_attention_hook(i, f'attn_layer_{i}') for i in range(st.session_state.num_layers)]
        
        for layer in range(st.session_state.num_layers):
            # wrap the _attn function to create logit attribution
            st.session_state.model.save_ctx[f'logit_layer_{layer}'] = dict()
            st.session_state.model.model.transformer[layer].self_attn._attn, old_fn= gpt_attn_wrapper(
                st.session_state.model.model.transformer[layer].self_attn._attn, 
                st.session_state.model.save_ctx[f'logit_layer_{layer}'], 
                st.session_state.model.model.transformer[layer].self_attn.o_proj.weight,
                st.session_state.model.model.embedding.weight.T,
                target_ids=target_ids,
            )
            
            st.session_state.model.forward(model_input, hooks=attn_hooks)
            
            # parse attentions
            attention = st.session_state.model.save_ctx[f"attn_layer_{layer}"]['attn'][0]
            attention = einops.rearrange(attention, 'h n1 n2 -> n1 n2 h')

            # parse logits
            if model_input.shape[1] > 1: # otherwise we don't have any logit attribution
                logits = st.session_state.model.save_ctx[f'logit_layer_{layer}']['logits']
                logits = torch.cat([torch.zeros_like(logits[:,0][:,None]), logits], dim=1)
                logits = torch.cat([logits, torch.zeros_like(logits[:,:,0][:,:,None])], dim=2)
                logits = einops.rearrange(logits, 'h n1 n2 -> n1 n2 h')
            else:
                logits = torch.zeros((attention.shape[0], attention.shape[1], attention.shape[2]))
                
            # compute and display the html object
            html_object = ps.AttentionLogits(tokens=tokenized_text, attention=attention, logits=logits, head_labels=[f'{layer}:{j}' for j in range(attention.shape[-1])])
            html_object = html_object.update_meta(suppress_title=True)
            html_str = html_object.html_page_str()
            
            # save html string
            save_destination[f'layer_{layer}'] = html_str
            
            # restore _attn functions
            st.session_state.model.model.transformer[layer].self_attn._attn = old_fn
    else:
        tokenized_text = st.session_state.tokenizer.tokenize(text)
        tokenized_text = [token.replace("Ġ", " ") for token in tokenized_text]
        tokenized_text = [token.replace("Ċ", "\n") for token in tokenized_text]
        model_input = st.session_state.tokenizer.encode(text, return_tensors='pt').to(st.session_state.device)
        target_ids = st.session_state.tokenizer.encode(text)[1:]
        
        # compute attention pattern
        attn_hooks = [gpt_get_attention_hook(i, f'attn_layer_{i}') for i in range(st.session_state.num_layers)]
        st.session_state.model.forward(model_input, hooks=attn_hooks, output_attentions=True)

        # compute logits
        for layer in range(st.session_state.num_layers):
            # wrap the _attn function to create logit attribution
            st.session_state.model.save_ctx[f'logit_layer_{layer}'] = dict()
            wrap_gpt_attn(layer, target_ids)

            # forward pass
            st.session_state.model.forward(model_input, hooks=[])
        
            # parse attentions for this layer
            attention = st.session_state.model.save_ctx[f"attn_layer_{layer}"]['attn'][0]
            attention = einops.rearrange(attention, 'h n1 n2 -> n1 n2 h')

            # parse logits
            if model_input.shape[1] > 1: # otherwise we don't have any logit attribution
                logits = st.session_state.model.save_ctx[f'logit_layer_{layer}']['logits']
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
            save_destination[f'layer_{layer}'] = html_str
            
            # reset _attn function
            reset_attn_fn(layer, old_fn)
            
            # save progress so far
            with open(st.session_state.model_name + ".json", "w") as f: 
                json.dump(st.session_state.visualization, f)

            # garbage collection
            gc.collect()
            torch.cuda.empty_cache()

def pad_logits(logits):
    logits = torch.cat([torch.zeros_like(logits[:,0][:,None]), logits], dim=1)
    logits = torch.cat([logits, torch.zeros_like(logits[:,:,0][:,:,None])], dim=2)
    return logits

def text_change(col_idx: Union[int, List[int]]):
    if isinstance(col_idx, list):
        for idx in col_idx:
            text_change(idx)
        return
    
    text = st.session_state["storage"][col_idx]
    if st.session_state.prefix_prompt is not None and len(st.session_state.prefix_prompt) > 0:
        text = st.session_state.prefix_prompt + '\n' + text

    if text is None or len(text) == 0:
        return
            
    compute_attn_logits(text, st.session_state.visualization[f'col_{col_idx}'])

def wrap_gpt_attn(layer, target_ids):
    st.session_state.model.m
    if hasattr(st.session_state.model.model.transformer.h[layer].attn, "_attn"):
        st.session_state.model.model.transformer.h[layer].attn._attn, old_fn= gpt_attn_wrapper(
            st.session_state.model.model.transformer.h[layer].attn._attn, 
            st.session_state.model.save_ctx[f'logit_layer_{layer}'], 
            st.session_state.model.model.transformer.h[layer].attn.c_proj.weight,
            st.session_state.model.model.lm_head.weight.T,
            target_ids=target_ids,
        )
    elif hasattr(st.session_state.model.model.transformer.h[layer].attn.attention, "_attn"):
        st.session_state.model.model.transformer.h[layer].attn.attention._attn, old_fn= gpt_attn_wrapper(
            st.session_state.model.model.transformer.h[layer].attn.attention._attn, 
            st.session_state.model.save_ctx[f'logit_layer_{layer}'], 
            st.session_state.model.model.transformer.h[layer].attn.attention.out_proj.weight,
            st.session_state.model.model.lm_head.weight.T,
            target_ids=target_ids,
        )
    else:
        AttributeError(f'Layer {layer} has no _attn function')
        

def reset_attn_fn(layer, old_fn):
    if hasattr(st.session_state.model.model.transformer.h[layer].attn, "_attn"):
        del st.session_state.model.model.transformer.h[layer].attn._attn
        st.session_state.model.model.transformer.h[layer].attn._attn = old_fn
    elif hasattr(st.session_state.model.model.transformer.h[layer].attn.attention, "_attn"):
        del st.session_state.model.model.transformer.h[layer].attn.attention._attn
        st.session_state.model.model.transformer.h[layer].attn.attention._attn = old_fn
    else:
        AttributeError(f'Layer {layer} has no _attn function')