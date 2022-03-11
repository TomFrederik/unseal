from typing import Union, List

import streamlit as st
from transformers.models.gpt_neo.modeling_gpt_neo import GPTNeoPreTrainedModel

from ..utils import compute_attn_logits

def sample_text(model, col_idx, key):
    text = st.session_state[key]
    if st.session_state.prefix_prompt is not None and len(st.session_state.prefix_prompt) > 0:
        text = st.session_state.prefix_prompt + '\n' + text
    model_inputs = st.session_state.tokenizer.encode(text, return_tensors='pt').to(st.session_state.device)
    output = model.model.generate(model_inputs, **st.session_state.sample_kwargs, min_length=0, output_attentions=True)
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

def get_attn_logits_args():
    # get args for compute_attn_logits
    if st.session_state.model_name in st.session_state.registered_model_names:
        attn_name = st.session_state.config['attn_name']
        output_idx = st.session_state.config['output_idx']
        layer_key_prefix = st.session_state.config['layer_key_prefix']
        out_proj_name = st.session_state.config['out_proj_name']
        attn_suffix = st.session_state.config['attn_suffix']
        unembedding_key = st.session_state.config['unembedding_key']
    elif isinstance(st.session_state.model.model, GPTNeoPreTrainedModel):
        attn_name = 'attn'
        output_idx = 2
        layer_key_prefix = 'transformer->h'
        out_proj_name = 'out_proj'
        attn_suffix = 'attention'
        unembedding_key = 'lm_head'
    else:
        attn_name = 'attn'
        output_idx = 2
        layer_key_prefix = 'transformer->h'
        out_proj_name = 'c_proj'
        attn_suffix = None
        unembedding_key = 'lm_head'
    return attn_name, output_idx, layer_key_prefix, out_proj_name, attn_suffix, unembedding_key

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
    
    attn_name, output_idx, layer_key_prefix, out_proj_name, attn_suffix, unembedding_key = get_attn_logits_args()
    
    compute_attn_logits(
        st.session_state.model, 
        st.session_state.model_name, 
        st.session_state.tokenizer,
        st.session_state.num_layers,
        text, 
        st.session_state.visualization[f'col_{col_idx}'],
        attn_name = attn_name,
        output_idx = output_idx,
        layer_key_prefix = layer_key_prefix,
        out_proj_name = out_proj_name,
        attn_suffix = attn_suffix,
        unembedding_key = unembedding_key,
    )