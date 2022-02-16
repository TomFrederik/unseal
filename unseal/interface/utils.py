import importlib
import json
from typing import List, Tuple, Optional

import streamlit as st
import torch

from unseal.hooks import HookedModel
from unseal.transformers_util import load_from_pretrained, get_num_layers

def init_session_state(variables: List[str]) -> None:
    """Initialize session state variables to None.

    :param variables: List of variable names to initialize.
    :type variables: List[str]
    """
    for var in variables:
        if var not in st.session_state:
            st.session_state[var] = None
            
def on_config_submit(model_name: str) -> Tuple:
    """Function that is called on submitting the config form.

    :param model_name: Name of the model that should be loaded
    :type model_name: str
    :return: Model, tokenizer, config
    :rtype: Tuple
    """
    if not torch.cuda.is_available():
        st.session_state.device = "cpu"
    # else: leave it as the selected device, so either cpu or cuda

    # load model, hook it and put it on device and in eval mode
    model, tokenizer, config = load_model(model_name)
    model = HookedModel(model)
    model.to(st.session_state.device).eval()
    
    st.session_state.num_layers = get_num_layers(model)
    print(st.session_state.num_layers)

    return model, tokenizer, config

@st.experimental_singleton
def load_model(model_name: str) -> Tuple:
    """Load the specified model with its tokenizer and config.

    :param model_name: Model name, e.g. 'gpt2-xl'
    :type model_name: str
    :return: Model, Tokenizer, Config
    :rtype: Tuple
    """
    if model_name in st.session_state.registered_model_names:
        # import model constructor
        constructor = st.session_state.registered_models[model_name]['constructor']
        constructor_module = importlib.import_module('.'.join(constructor.split('.')[:-1]))
        constructor_class = getattr(constructor_module, constructor.split('.')[-1])
        
        # load model from checkpoint --> make sure that your class has this method, it's default for pl.LightningModules
        checkpoint = st.session_state.registered_models[model_name]['checkpoint'] 
        model = constructor_class.load_from_checkpoint(checkpoint)

        # load tokenizer
        tokenizer = st.session_state.registered_models[model_name]['tokenizer'] # TODO how to deal with this?
        tokenizer_module = importlib.import_module('.'.join(tokenizer.split('.')[:-1]))
        tokenizer_class = getattr(tokenizer_module, tokenizer.split('.')[-1])
        tokenizer = tokenizer_class()

        # TODO?
        config = None

    else: # attempt to load from huggingface
        model, tokenizer, config = load_from_pretrained(model_name)

    return model, tokenizer, config

def load_registered_models(model_file_path: str = './registered_models.json') -> None:
    try:
        with open(model_file_path, 'r') as f:
            st.session_state.registered_models = json.load(f)
    except FileNotFoundError:
        st.warning(f"Did not find a 'registered_models.json'. Only showing HF models")
        st.session_state.registered_models = dict()
    st.session_state.registered_model_names = list(st.session_state.registered_models.keys())

def startup(variables: List[str], mode_file_path: Optional[str] = './registered_models.json') -> None:
    """Performs startup tasks for the app.

    :param variables: List of variable names that should be intialized.
    :type variables: List[str]
    :param model_file_path: Path to the file containing the registered models.
    :type model_file_path: Optional[str]
    """
    # set wide layout
    st.set_page_config(layout="wide")
    
    # initialize session state variables
    init_session_state(variables)

    # load externally registered models
    load_registered_models(mode_file_path)
