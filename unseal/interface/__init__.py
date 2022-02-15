from . import utils

# define some global variables
HF_MODELS = [
    'gpt2', 
    'gpt2-medium', 
    'gpt2-large', 
    'gpt2-xl'
]

SESSION_STATE_VARIABLES = [
    'model',
    'tokenizer',
    'config',
    'input_text',
    'registered_models',
    'registered_model_names',
    'num_layers'
]