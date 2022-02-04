import argparse
import json
import os

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPTNeoForCausalLM

from info_flow import eval_model
from hooks import FullModelHooks

NUM_LAYERS = {
    'gpt2': {'small':12, 'medium':24, 'large':36, 'xl':48},
    'gpt-neo': {'125m':12, '1.3b':24, '2.7b':32},
}

SIZE2SUFFIX = {
    'gpt-neo': {'125m':'125M', '1.3b':'1.3B', '2.7b':'2.7B'},
    'gpt2': {'small':'', 'medium':'-medium', 'large':'-large', 'xl':'-xl'},
}

def main(args):
    device = 'cuda' if (torch.cuda.is_available() and args.model_size not in ['xl', '2.7b']) else 'cpu' # larger doesn't fit on my gpu

    if args.model == 'gpt2':
        tokenizer = GPT2Tokenizer.from_pretrained(f"gpt2{SIZE2SUFFIX[args.model][args.model_size]}")
        model = GPT2LMHeadModel.from_pretrained(f"gpt2{SIZE2SUFFIX[args.model][args.model_size]}")
    elif args.model == 'gpt-neo':
        tokenizer = GPT2Tokenizer.from_pretrained(f"EleutherAI/gpt-neo-{SIZE2SUFFIX[args.model][args.model_size]}")
        model = GPTNeoForCausalLM.from_pretrained(f"EleutherAI/gpt-neo-{SIZE2SUFFIX[args.model][args.model_size]}")
    else:
        raise ValueError(f'Unrecognized model {args.model}')
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

        results = eval_model(model, tokenizer, base_text, entity, correct_output_text, NUM_LAYERS[args.model][args.model_size], hooks)
        results['prompt'] = base_text
        results['correct'] = correct_output_text
        results['entity'] = entity
        all_results[i] = results
    
    os.makedirs(f'./info_results/{args.model}/{args.model_size}', exist_ok=True)
    with open(f'./info_results/{args.model}/{args.model_size}/results.json', 'w') as f:
        json.dump(all_results, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['gpt2', 'gpt-neo'], default='gpt2')
    parser.add_argument('--model_size', type=str, help='Model size, e.g. large or xl for gpt2 or 125m for gpt-neo')
    parser.add_argument('--text_file', default='prompts.json', help='File that contains the data')

    args = parser.parse_args()
    
    main(args)