import json
import time

import streamlit as st

from unseal.visuals.streamlit_interfaces import utils
from unseal.visuals.streamlit_interfaces import interface_setup as setup
from unseal.visuals.streamlit_interfaces.commons import SESSION_STATE_VARIABLES

# perform startup tasks
setup.startup(SESSION_STATE_VARIABLES, './registered_models.json')

# create sidebar
with st.sidebar:
    setup.create_sidebar()
    
    sample = st.checkbox('Enable sampling', value=False, key='sample')
    if sample:
        setup.create_sample_sliders()
        setup.on_sampling_config_change()
    
    if "storage" not in st.session_state:
        st.session_state["storage"] = [""]
        
    # input 1
    placeholder1 = st.empty()
    placeholder1.text_area(label='Input 1', on_change=utils.on_text_change, key='input_text_1', value=st.session_state.storage[0], kwargs=dict(col_idx=0, text_key='input_text_1'))
    if sample:
        st.button(label="Sample", on_click=utils.sample_text, kwargs=dict(model=st.session_state.model, col_idx=0, key="input_text"), key="sample_text")
    
    # sometimes need to force a re-render
    st.button('Show Attention', on_click=utils.text_change, kwargs=dict(col_idx=0))
    
    f =  json.encoder.JSONEncoder().encode(st.session_state.visualization)
    st.download_button(
        label='Download Visualization', 
        data=f, 
        file_name=f'{st.session_state.model_name}_{time.strftime("%Y%m%d_%H%M%S", time.localtime())}.json', 
        mime='application/json', 
        help='Download the visualizations as a json of html files.', 
        key='download_button'
    )

# show the html visualization
if st.session_state.model is not None:
    cols = st.columns(1)
    for col_idx, col in enumerate(cols):
        if f"col_{col_idx}" in st.session_state.visualization:
            with col:
                for layer in range(st.session_state.num_layers):
                    with st.expander(f'Layer {layer}'):
                        st.components.v1.html(st.session_state.visualization[f"col_{col_idx}"][f"layer_{layer}"], height=600)
