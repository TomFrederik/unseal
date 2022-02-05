import argparse
from itertools import product
import json
import os

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPTNeoForCausalLM

import transformers_util as tutil

def main(args):

    # load tokenizer and config
    model_name = args.model
    if args.model_size is not None:
        model_name += '-' + args.model_size
    else:
        args.model_size = 'None'

    _, tokenizer, config = tutil.load(model_name, args.model_dir, load_model=False)

    # get num layers from config
    num_layers = config.num_hidden_layers

    # load data
    model_dir = os.path.join(args.results_dir, args.model, args.model_size) 
    with open(os.path.join(model_dir, 'results.json'), 'r') as f:
        data = json.load(f)

    for i, exp in data.items():
        
        startstop = exp['entity'].split(':')
        stop = int(startstop[-1])
        
        if len(startstop) > 1 and startstop[0] != '':
            start = int(startstop[0]) # TODO, currently only supports contiguous entities
        else:
            start = 0


        tokens = tokenizer(exp['prompt'], return_tensors='pt')['input_ids'][0][:,None]
        tokenized_prompt = tokenizer.batch_decode(tokens)
        tokenized_prompt = [p.lstrip(' ') for p in tokenized_prompt]
        tokenized_prompt[start:stop] = [token + '*' for token in tokenized_prompt[start:stop]]
        correct = tokenized_prompt[-1]
        tokenized_prompt = tokenized_prompt[:-1]
        num_tokens = len(tokenized_prompt)

        prob_array = np.zeros((num_tokens,num_layers))

        for n, pos in product(range(num_layers), range(num_tokens)):
            prob_array[pos, n] = exp['hidden'][str(n)][str(pos)]

        plt.figure(figsize=(10,6))
        plt.title(f"{args.model}-{args.model_size}")
        plt.xticks(np.arange(0,num_layers,5)+0.5, np.arange(0,num_layers,5))
        plt.yticks(np.arange(0,num_tokens)+0.5, tokenized_prompt)
        im = plt.pcolormesh(prob_array, cmap="Purples")
        plt.gca().invert_yaxis()
        cbar = plt.colorbar(im)
        cbar.ax.set_title(f"p({correct})", y=-0.07)
        plt.savefig(os.path.join(model_dir, f'{correct}_hidden.png'))


        #########

        prob_array = np.zeros((num_tokens,num_layers))

        for n, pos in product(range(num_layers), range(num_tokens)):
            prob_array[pos, n] = exp['mlp'][str(n)][str(pos)]

        plt.figure(figsize=(10,6))
        plt.title(f"{args.model}-{args.model_size}")
        plt.xticks(np.arange(0,num_layers,5)+0.5, np.arange(0,num_layers,5))
        plt.yticks(np.arange(0,num_tokens)+0.5, tokenized_prompt)
        im = plt.pcolormesh(prob_array, cmap="Greens")
        plt.gca().invert_yaxis()
        cbar = plt.colorbar(im)
        cbar.ax.set_title(f"p({correct})", y=-0.07)
        plt.savefig(os.path.join(model_dir, f'{correct}_mlp.png'))

        #########

        prob_array = np.zeros((num_tokens,num_layers))

        for n, pos in product(range(num_layers), range(num_tokens)):
            prob_array[pos, n] = exp['attn'][str(n)][str(pos)]

        plt.figure(figsize=(10,6))
        plt.title(f"{args.model}-{args.model_size}")
        plt.xticks(np.arange(0,num_layers,5)+0.5, np.arange(0,num_layers,5))
        plt.yticks(np.arange(0,num_tokens)+0.5, tokenized_prompt)
        im = plt.pcolormesh(prob_array, cmap="Reds")
        plt.gca().invert_yaxis()
        cbar = plt.colorbar(im)
        cbar.ax.set_title(f"p({correct})", y=-0.07)
        plt.savefig(os.path.join(model_dir, f'{correct}_attn.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='gpt2')
    parser.add_argument('--model_size', type=str, help='Model size, e.g. large or xl for gpt2 or 125M for gpt-neo', default=None)
    parser.add_argument('--model_dir', default=None, help='HF directory in which to look for the specified model, e.g. EleutherAI')
    parser.add_argument('--results_dir', default='./info_results')
    
    args = parser.parse_args()
    
    main(args)