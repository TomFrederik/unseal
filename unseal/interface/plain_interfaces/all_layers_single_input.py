import streamlit as st
import torch

from unseal.interface import utils
from unseal.interface.commons import SESSION_STATE_VARIABLES

def layer_change():
    utils.text_change()


# perform startup tasks
utils.startup(SESSION_STATE_VARIABLES, './registered_models.json')

with st.sidebar:
    st.checkbox('Show only local models', value=False, key='local_only')
    if not st.session_state.local_only:
        model_names = st.session_state.registered_model_names + HF_MODELS
    else:
        model_names = st.session_state.registered_model_names
    
    with st.form('model_config'):
        
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

        devices = ['cpu']
        if torch.cuda.is_available():
            devices += ['cuda']
        st.selectbox(
            'Device',
            options=devices,
            index=0,
            key='device'
        )
        
        st.text_area(label='Prefix Prompt', key='prefix_prompt', value='')
        st.text_area(label='Suffix Prompt', key='suffix_prompt', value='')
            
        submitted = st.form_submit_button("Save config")
        if submitted:
            st.session_state.model, st.session_state.tokenizer, st.session_state.config = utils.on_config_submit(st.session_state.model_name)
            st.write('Config saved!')
    #TODO: text change doesn't work for single input
    input_text = st.text_area(label='Input', on_change=text_change, key='input_text', value="")

    st.button('Show Attention', on_click=text_change)