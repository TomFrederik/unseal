import importlib
import json
from typing import List, Tuple, Optional

import einops
import pysvelte as ps
import streamlit as st
import torch

from ..hooks import HookedModel
from ..hooks.common_hooks import gpt_get_attention_hook
from ..transformers_util import load_from_pretrained, get_num_layers
from .commons import HF_MODELS

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
    # load model, hook it and put it on device and in eval mode
    model, tokenizer, config = load_model(model_name)
    model = HookedModel(model)
    model.to(st.session_state.device).eval()
    
    st.session_state.num_layers = get_num_layers(model)

    return model, tokenizer, config

def on_sampling_config_change():
    st.session_state.sample_kwargs = dict(
        temperature=st.session_state.temperature,
        max_length=st.session_state.response_length,
        top_p=st.session_state.top_p,
        repetition_penalty=1/st.session_state.repetition_penalty,
        num_beams=st.session_state.num_beams
    )

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

def sample_text(storage_key, label, key):
    text = st.session_state[key]
    if st.session_state.prefix_prompt is not None and len(st.session_state.prefix_prompt) > 0:
        text = st.session_state.prefix_prompt + '\n' + text
    model_inputs = st.session_state.tokenizer.encode(text, return_tensors='pt').to(st.session_state.device)
    output = st.session_state.model.model.generate(model_inputs, **st.session_state.sample_kwargs, min_length=0, output_attentions=True)
    output_text = st.session_state.tokenizer.decode(output[0], skip_special_tokens=True)
    if st.session_state.prefix_prompt is not None and len(st.session_state.prefix_prompt) > 0:
        output_text = output_text.lstrip(st.session_state.prefix_prompt + '\n')
    
    st.session_state[storage_key] = output_text
    text_change()

def create_sample_sliders():
    st.slider(label="Temperature", min_value=0., max_value=1.0, value=0., step=0.01, key='temperature', on_change=on_sampling_config_change)
    st.slider(label="Response length", min_value=1, max_value=1024, value=64, step=1, key='response_length', on_change=on_sampling_config_change)
    st.slider(label="Top P", min_value=0., max_value=1.0, value=1., step=0.01, key='top_p', on_change=on_sampling_config_change)
    st.slider(label="Repetition Penalty (1 = no penalty)", min_value=0.01, max_value=1.0, value=1., step=0.01, key='repetition_penalty', on_change=on_sampling_config_change)
    st.slider(label="Number of Beams", min_value=1, max_value=10, value=1, step=1, key='num_beams', on_change=on_sampling_config_change)

def on_text_change(storage_key, text_key):
    st.session_state[storage_key] = st.session_state[text_key]
    text_change()
    
def text_change():
    cols = st.columns(2)
    
    for k, text in enumerate([st.session_state.storage_1, st.session_state.storage_2]):
        with cols[k]:
            if st.session_state.prefix_prompt is not None and len(st.session_state.prefix_prompt) > 0:
                text = st.session_state.prefix_prompt + '\n' + text
            # if st.session_state.suffix_prompt is not None and len(st.session_state.suffix_prompt) > 0:
            #     text = text + '\n' + st.session_state.suffix_prompt

            if text is None or len(text) == 0:
                return
            if st.session_state.model_name.endswith('grokking'):
                raise NotImplementedError
            else:
                tokenized_text = st.session_state.tokenizer.tokenize(text)
                tokenized_text = [token.replace("Ġ", " ") for token in tokenized_text]
                tokenized_text = [token.replace("Ċ", "\n") for token in tokenized_text]
                model_input = st.session_state.tokenizer.encode(text, return_tensors='pt').to(st.session_state.device)

                layer_hooks = [gpt_get_attention_hook(i, f'layer_{i}') for i in range(st.session_state.num_layers)]
                st.session_state.model.forward(model_input, hooks=layer_hooks, output_attentions=True)
                layer_attentions = [st.session_state.model.save_ctx[f'layer_{i}']['attn'] for i in range(st.session_state.num_layers)]
                layer_attentions = [einops.rearrange(attn[0], 'h n1 n2 -> n1 n2 h') for attn in layer_attentions]
            
            for i, attn in enumerate(layer_attentions):
                html_object = ps.AttentionMulti(tokens=tokenized_text, attention=attn, head_labels=[f'{i}:{j}' for j in range(attn.shape[-1])])
                html_object = html_object.update_meta(suppress_title=True)
                html_str = html_object.html_page_str()
                with st.expander(f'Layer {i}'):
                    st.components.v1.html(html_str, height=600)

def create_model_config(model_names):
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
            st.session_state.model, st.session_state.tokenizer, st.session_state.config = on_config_submit(st.session_state.model_name)
            st.write('Model config saved!')


def create_sidebar():
    st.checkbox('Show only local models', value=False, key='local_only')
    
    if not st.session_state.local_only:
        model_names = st.session_state.registered_model_names + HF_MODELS
    else:
        model_names = st.session_state.registered_model_names
    
    create_model_config(model_names)