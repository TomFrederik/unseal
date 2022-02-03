import argparse
import json
import os

import torch
from transformers import GPT2Tokenizer, GPTNeoForCausalLM
from tqdm import tqdm

from ROME import eval_model
from hooks import FullModelHooks

SIZE2SUFFIX = {'125m':'125M', '1.3b':'1.3B', '2.7b':'2.7B'}
NUM_LAYERS = {'125m':12, '1.3b':24, '2.7b':32}


def main(args):
    device = 'cuda' if (torch.cuda.is_available() or args.model_size != '125m') else 'cpu' # larger doesn't fit on my gpu


    tokenizer = GPT2Tokenizer.from_pretrained(f"EleutherAI/gpt-neo-{SIZE2SUFFIX[args.model_size]}")
    model = GPTNeoForCausalLM.from_pretrained(f"EleutherAI/gpt-neo-{SIZE2SUFFIX[args.model_size]}")
    model.to(device)
    model.eval()

    hooks = FullModelHooks(model)

    with open(args.text_file, "r") as f:
        data = json.load(f)
    
    prompts = data['prompts']
    corrects = data['correct']
    entities = data['entity']
    
    all_results = dict()
    for i in tqdm(range(len(prompts))):
        base_text = prompts[i]
        correct_output_text = corrects[i]
        entity = entities[i]

        results = eval_model(model, tokenizer, base_text, entity, correct_output_text, NUM_LAYERS[args.model_size], hooks)
        results['prompt'] = base_text
        results['correct'] = correct_output_text
        results['entity'] = entity
        all_results[i] = results
    
    os.makedirs(f'./rome_results/gpt-neo/{args.model_size}', exist_ok=True)
    with open(f'./rome_results/gpt-neo/{args.model_size}/results.json', 'w') as f:
        json.dump(all_results, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_size', choices=['125m', '1.3b', '2.7b'], default='125m')
    parser.add_argument('--text_file', default='prompts.json')

    args = parser.parse_args()
    
    main(args)