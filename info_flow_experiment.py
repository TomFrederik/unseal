import argparse
import json
import logging
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.file_utils import RepositoryNotFoundError

from info_flow import eval_model
import hooks

def main(args):
    device = 'cpu' # for debugging and testing larger models
    # device = 'cuda' if (torch.cuda.is_available() and args.model_size not in ['xl', '2.7B']) else 'cpu' # larger doesn't fit on my gpu

    # assemble model name
    model_name = args.model
    if args.model_size is not None:
        model_name += '-' + args.model_size

    # Trying to load model, tokenizer and config
    try:
        logging.info(f'Loading model {model_name}')

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        config = AutoConfig.from_pretrained(model_name)

    except (RepositoryNotFoundError, OSError) as error:
        logging.warning("Couldn't find model in default folder. Trying EleutherAI/...")

        tokenizer = AutoTokenizer.from_pretrained(f'EleutherAI/{model_name}')
        model = AutoModelForCausalLM.from_pretrained(f'EleutherAI/{model_name}')
        config = AutoConfig.from_pretrained(f'EleutherAI/{model_name}')
    
    model.to(device)
    model.eval()
    model = hooks.HookedModel(model)

    num_layers = config.num_hidden_layers
    logging.info(f'{num_layers = }')

    os.makedirs(f'./info_results/{args.model}/{args.model_size}', exist_ok=True)

    with open(args.text_file, "r") as f:
        data = json.load(f)
    
    prompts = data['prompts']
    corrects = data['correct']
    entities = data['entity']
    
    all_results = dict()
    for i in range(len(prompts)):
        base_text = prompts[i]
        correct_output_text = corrects[i]
        entity = entities[i]

        results = eval_model(model, tokenizer, base_text, entity, correct_output_text, num_layers)
        results['prompt'] = base_text
        results['correct'] = correct_output_text
        results['entity'] = entity
        all_results[i] = results
    
        with open(f'./info_results/{args.model}/{args.model_size}/results.json', 'w') as f:
            json.dump(all_results, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='gpt2')
    parser.add_argument('--model_size', type=str, help='Model size, e.g. large or xl for gpt2 or 125M for gpt-neo', default=None)
    parser.add_argument('--text_file', default='prompts.json', help='File that contains the data')

    args = parser.parse_args()
    
    main(args)