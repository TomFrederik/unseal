import streamlit as st
import torch

from unseal.interface import utils, HF_MODELS, SESSION_STATE_VARIABLES

# perform startup tasks
utils.startup(SESSION_STATE_VARIABLES, './registered_models.json')

# create sidebar
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
        # st.text_area(label='Suffix Prompt', key='suffix_prompt', value='')
            
        submitted = st.form_submit_button("Save model config")
        if submitted:
            st.session_state.model, st.session_state.tokenizer, st.session_state.config = utils.on_config_submit(st.session_state.model_name)
            st.write('Model config saved!')
    
    sample = st.checkbox('Enable sampling', value=False, key='sample')
    if sample:
        utils.create_sample_sliders()
        utils.on_sampling_config_change()
    
    if "storage_1" not in st.session_state:
        st.session_state["storage_1"] = ""
        st.session_state["storage_2"] = ""
        
    # input 1
    placeholder1 = st.empty()
    placeholder1.text_area(label='Input 1', on_change=utils.on_text_change, key='input_text_1', value=st.session_state.storage_1, kwargs=dict(storage_key="storage_1", text_key='input_text_1'))
    if sample:
        st.button(label="Sample", on_click=utils.sample_text, kwargs=dict(storage_key="storage_1", label='Input 1', key="input_text_1"), key="sample_text_1")
    
    # input 2
    placeholder2 = st.empty()
    placeholder2.text_area(label='Input 2', on_change=utils.on_text_change, key='input_text_2', value=st.session_state.storage_2, kwargs=dict(storage_key="storage_2", text_key='input_text_2'))
    if sample:
        st.button(label="Sample", on_click=utils.sample_text, kwargs=dict(storage_key="storage_2", label='Input 2', key="input_text_2"), key="sample_text_2")

    # sometimes need to force a re-render
    st.button('Show Attention', on_click=utils.text_change)


