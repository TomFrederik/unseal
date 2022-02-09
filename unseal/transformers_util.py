import logging
import os

from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from transformers.file_utils import RepositoryNotFoundError


def load(model_name, model_dir=None, load_model=True, load_tokenizer=True, load_config=True):
    if model_dir is None:
        try:
            logging.info(f'Loading model {model_name}')

            model = AutoModelForCausalLM.from_pretrained(model_name) if load_model else None        
            tokenizer = AutoTokenizer.from_pretrained(model_name) if load_tokenizer else None
            config = AutoConfig.from_pretrained(model_name) if load_config else None

        except (RepositoryNotFoundError, OSError) as error:
            logging.warning("Couldn't find model in default folder. Trying EleutherAI/...")

            model = AutoModelForCausalLM.from_pretrained(f'EleutherAI/{model_name}') if load_model else None
            tokenizer = AutoTokenizer.from_pretrained(f'EleutherAI/{model_name}') if load_tokenizer else None
            config = AutoConfig.from_pretrained(f'EleutherAI/{model_name}') if load_config else None

    else:
        model = AutoModelForCausalLM.from_pretrained(os.path.join(model_dir, model_name)) if load_model else None        
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_dir, model_name)) if load_tokenizer else None
        config = AutoConfig.from_pretrained(os.path.join(model_dir, model_name)) if load_config else None

    return model, tokenizer, config