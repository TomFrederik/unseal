
import einops
import pysvelte as ps
import streamlit as st
import torch
from unseal.transformers_util import load_from_pretrained
from unseal.hooks import HookedModel
from unseal.hooks.common_hooks import gpt_get_attention_hook

@st.experimental_singleton
def init(model_name):
    # st.session_state.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    st.session_state.device = "cpu"
    
    if model_name == 'Other (specify below)':
        raise NotImplementedError()
    else:
        model, tokenizer, config = load_from_pretrained(model_name)
    
    model = HookedModel(model)
    model.to(st.session_state.device).eval()

    return model, tokenizer, config

def layer_change():
    st.session_state.attn_hook =  gpt_get_attention_hook(st.session_state.layer, 'my_attn')

if 'model' not in st.session_state:
    st.session_state['model'] = None
if 'tokenizer' not in st.session_state:
    st.session_state['tokenizer'] = None
if 'config' not in st.session_state:
    st.session_state['config'] = None
if 'attn_hook' not in st.session_state:
    st.session_state['attn_hook'] = None
if 'input_text' not in st.session_state:
    st.session_state['input_text'] = None
if 'device' not in st.session_state:
    st.session_state['device'] = None

def text_change():
    text = st.session_state.input_text
    if text is None or len(text) == 0:
        return

    tokenized_text = st.session_state.tokenizer.tokenize(text)
    tokenized_text = [token.replace("Ġ", " ") for token in tokenized_text]
    tokenized_text = [token.replace("Ċ", "\n") for token in tokenized_text]
    model_input = st.session_state.tokenizer.encode(text, return_tensors='pt').to(st.session_state.device)

    hook = gpt_get_attention_hook(st.session_state.layer, 'my_hook')
    st.session_state.model.forward(model_input, hooks=[hook], output_attentions=True)
    attn = st.session_state.model.save_ctx['my_hook']['attn']
    attn = einops.rearrange(attn[0], 'h n1 n2 -> n1 n2 h')
    
    
    html_object = ps.AttentionMulti(tokens=tokenized_text, attention=attn, head_labels=[f'{st.session_state.layer}:{i}' for i in range(attn.shape[-1])])
    html_str = html_object.html_page_str()
    print(html_str)
    st.components.v1.html(html_str, height=1200)
        

with st.sidebar:
    with st.form('site_config'):
        st.write('## Config')
        
        st.selectbox(
            'Model', 
            options=['gpt2', 'gpt2-large'],
            key='model_name',
            index=0,
        )
            
        submitted = st.form_submit_button("Save config")
        if submitted:
            st.session_state.model, st.session_state.tokenizer, st.session_state.config = init(st.session_state.model_name)
            st.write('Config saved!')

    st.selectbox('layer', options=list(range(12)), key='layer', on_change=layer_change, index=0)

    input_text = st.text_area(label='Input', on_change=text_change, key='input_text')
