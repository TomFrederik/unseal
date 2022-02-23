import time
import json
import streamlit as st

from unseal.interface import utils
from unseal.interface.commons import SESSION_STATE_VARIABLES

# perform startup tasks
if 'startup_done' not in st.session_state:
    st.session_state['startup_done'] = False
    utils.startup(SESSION_STATE_VARIABLES, './registered_models.json')

# create sidebar
with st.sidebar:
    utils.create_sidebar()
    
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
    
    f =  json.encoder.JSONEncoder().encode(st.session_state.visualization)
    st.download_button(
        label='Download Visualization', 
        data=f, 
        file_name=f'{st.session_state.model_name}_{time.strftime("%Y%m%d_%H%M%S", time.localtime())}.json', 
        mime='application/json', 
        help='Download the visualizations as a json of html files.', 
        key='download_button'
    )
