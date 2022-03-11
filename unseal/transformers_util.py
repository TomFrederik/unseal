# utility functions for interacting with huggingface's transformers library
import logging
import os
from typing import Optional, Tuple

from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from transformers.file_utils import RepositoryNotFoundError

from .hooks.commons import HookedModel

def load_from_pretrained(
    model_name: str, 
    model_dir: Optional[str] = None, 
    load_model: Optional[bool] = True, 
    load_tokenizer: Optional[bool] = True, 
    load_config: Optional[bool] = True,
    low_cpu_mem_usage: Optional[bool] = False,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer, AutoConfig]:
    """Load a pretrained model from huggingface's transformer library

    :param model_name: Name of the model, e.g. `gpt2` or `gpt-neo`.
    :type model_name: str
    :param model_dir: Directory in which to look for the model, e.g. `EleutherAI`, defaults to None
    :type model_dir: Optional[str], optional
    :param load_model: Whether to load the model itself, defaults to True
    :type load_model: Optional[bool], optional
    :param load_tokenizer: Whether to load the tokenizer, defaults to True
    :type load_tokenizer: Optional[bool], optional
    :param load_config: Whether to load the config file, defaults to True
    :type load_config: Optional[bool], optional
    :param low_cpu_mem_usage: Whether to use low-memory mode, experimental feature of HF, defaults to False
    :type low_cpu_mem_usage: bool, optional
    :return: model, tokenizer, config. Returns None values for those elements which were not loaded.
    :rtype: Tuple[AutoModelForCausalLM, AutoTokenizer, AutoConfig]
    """
    if model_dir is None:
        try:
            logging.info(f'Loading model {model_name}')

            model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=low_cpu_mem_usage) if load_model else None        
            tokenizer = AutoTokenizer.from_pretrained(model_name) if load_tokenizer else None
            config = AutoConfig.from_pretrained(model_name) if load_config else None

        except (RepositoryNotFoundError, OSError) as error:
            logging.warning("Couldn't find model in default folder. Trying EleutherAI/...")

            model = AutoModelForCausalLM.from_pretrained(f'EleutherAI/{model_name}', low_cpu_mem_usage=low_cpu_mem_usage) if load_model else None
            tokenizer = AutoTokenizer.from_pretrained(f'EleutherAI/{model_name}') if load_tokenizer else None
            config = AutoConfig.from_pretrained(f'EleutherAI/{model_name}') if load_config else None

    else:
        model = AutoModelForCausalLM.from_pretrained(os.path.join(model_dir, model_name), low_cpu_mem_usage=low_cpu_mem_usage) if load_model else None        
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_dir, model_name)) if load_tokenizer else None
        config = AutoConfig.from_pretrained(os.path.join(model_dir, model_name)) if load_config else None

    return model, tokenizer, config

def get_num_layers(model: HookedModel, layer_key_prefix: Optional[str] = None) -> int:
    """Get the number of layers in a model

    :param model: The model to get the number of layers from
    :type model: HookedModel
    :return: The number of layers in the model
    :rtype: int
    """
    if layer_key_prefix is None:
        layer_key_prefix = ""
    
    return len(model.layers[f"{layer_key_prefix}"])