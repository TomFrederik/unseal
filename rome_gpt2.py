import argparse
import json
import os

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from ROME import eval_model
from hooks import FullModelHooks

SIZE2SUFFIX = {'small':'', 'medium':'-medium', 'large':'-large', 'xl':'-xl'}
NUM_LAYERS = {'small':12, 'medium':24, 'large':36, 'xl':48}


def main(args):
    device = 'cuda' if (torch.cuda.is_available() and args.model_size != 'xl') else 'cpu' # xl doesn't fit on my gpu


    tokenizer = GPT2Tokenizer.from_pretrained(f"gpt2{SIZE2SUFFIX[args.model_size]}")
    model = GPT2LMHeadModel.from_pretrained(f"gpt2{SIZE2SUFFIX[args.model_size]}")
    model.to(device)
    model.eval()

    hooks = FullModelHooks(model)

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

        results = eval_model(model, tokenizer, base_text, entity, correct_output_text, NUM_LAYERS[args.model_size], hooks)
        results['prompt'] = base_text
        results['correct'] = correct_output_text
        results['entity'] = entity
        all_results[i] = results
    
    os.makedirs(f'./rome_results/gpt2/{args.model_size}', exist_ok=True)
    with open(f'./rome_results/gpt2/{args.model_size}/results.json', 'w') as f:
        json.dump(all_results, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_size', choices=['small', 'medium', 'large', 'xl'], default='large')
    parser.add_argument('--text_file', default='prompts.json')

    args = parser.parse_args()
    
    main(args)