import einops
import pysvelte as ps
import streamlit as st
import torch

from unseal.hooks import Hook
from unseal.hooks.common_hooks import gpt_get_attention_hook
from unseal.interface import utils, HF_MODELS, SESSION_STATE_VARIABLES

def layer_change():
    text_change()

def text_change():
    if st.session_state.prefix_prompt is not None and len(st.session_state.prefix_prompt) > 0:
        text = st.session_state.prefix_prompt + '\n' + st.session_state.input_text
    else:
        text = st.session_state.input_text
    if st.session_state.suffix_prompt is not None and len(st.session_state.suffix_prompt) > 0:
        text = text + '\n' + st.session_state.suffix_prompt

    if text is None or len(text) == 0:
        return
    if st.session_state.model_name.endswith('grokking'):
        tokenized_text = text.split(' ')
        tokenized_text[1] = st.session_state.model.model.hparams.num_tokens - 2
        tokenized_text[3] = st.session_state.model.model.hparams.num_tokens - 1
        model_input = torch.Tensor([int(t) for t in tokenized_text])[None].to(st.session_state.device).long()
        def func(save_ctx, input, output):
            save_ctx['attn'] = output[1].detach().cpu()
        hook = Hook(f"transformer->{st.session_state.layer}->self_attn", func, 'my_hook')
        st.session_state.model.forward(model_input, hooks=[hook])
        attn = st.session_state.model.save_ctx['my_hook']['attn']
        tokenized_text = [str(t) for t in tokenized_text]
        attn = einops.rearrange(attn[0], 'h n1 n2 -> n1 n2 h') 
    else:
        tokenized_text = st.session_state.tokenizer.tokenize(text)
        tokenized_text = [token.replace("Ġ", " ") for token in tokenized_text]
        tokenized_text = [token.replace("Ċ", "\n") for token in tokenized_text]
        model_input = st.session_state.tokenizer.encode(text, return_tensors='pt').to(st.session_state.device)

        hook = gpt_get_attention_hook(st.session_state.layer, 'my_hook')
        st.session_state.model.forward(model_input, hooks=[hook], output_attentions=True)
        attn = st.session_state.model.save_ctx['my_hook']['attn']
        attn = einops.rearrange(attn[0], 'h n1 n2 -> n1 n2 h')
        
    
    html_object = ps.AttentionMulti(tokens=tokenized_text, attention=attn, head_labels=[f'{st.session_state.layer}:{i}' for i in range(attn.shape[-1])])
    html_object.update_meta(suppress_title=True)
    html_str = html_object.html_page_str()
    st.components.v1.html(html_str, height=1200)

# perform startup tasks
utils.startup(SESSION_STATE_VARIABLES, './registered_models.json')

with st.sidebar:
    st.checkbox('Show only local models', value=False, key='local_only')

    if not st.session_state.local_only:
        model_names = st.session_state.registered_model_names + HF_MODELS
    else:
        model_names = st.session_state.registered_model_names
    
    with st.form('model_config'):
        submitted = st.form_submit_button("Save config")
        
        st.write('## Model Config')

        if model_names is None:
            model_options = list()
        else:
            model_options = model_names
        st.selectbox(
            'Model', 
            options=model_options,
            key='model_name',
            index=0,
        )

        st.selectbox(
            'Device',
            options=['cpu', 'cuda'],
            index=0,
            key='device'
        )
        
        st.text_area(label='Prefix Prompt', key='prefix_prompt', value='')
        st.text_area(label='Suffix Prompt', key='suffix_prompt', value='')
            
        if submitted:
            st.session_state.model, st.session_state.tokenizer, st.session_state.config = utils.on_config_submit(st.session_state.model_name)
            st.write('Config saved!')

    if st.session_state.num_layers is None:
        options = list()
    else:
        options = list(range(st.session_state.num_layers))
    st.selectbox('Layer', options=options, key='layer', on_change=layer_change, index=0)

    input_text = st.text_area(label='Input', on_change=text_change, key='input_text', value="")

    st.button('Show Attention', on_click=text_change)