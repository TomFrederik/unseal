import argparse
from itertools import product
import json
import os

import matplotlib.pyplot as plt
from matplotlib import colors

import numpy as np

def main(args):
    model_dir = os.path.join(args.results_dir, args.model, args.model_size) 
    
    with open(os.path.join(model_dir, 'results.json'), 'r') as f:
        data = json.load(f)

    for i, exp in data.items():
        num_tokens = len(exp['mlp']['0'])
        num_layers = len(exp['mlp'])
        correct = exp['correct'][1:]

        prob_array = np.zeros((num_tokens,num_layers))

        for num_layers, pos in product(range(num_layers), range(num_tokens)):
            prob_array[pos, num_layers] = exp['hidden'][str(num_layers)][str(pos)]
        
        plt.figure(figsize=(10,6))
        plt.xticks(np.arange(0,num_layers,5)+0.5, np.arange(0,num_layers,5))
        plt.yticks()
        plt.yticks(np.arange(0,num_tokens)+0.5)
        im = plt.pcolormesh(prob_array, cmap="Purples")
        plt.gca().invert_yaxis()
        cbar = plt.colorbar(im)
        cbar.ax.set_title(f"p({correct})", y=-0.07)
        plt.savefig(os.path.join(model_dir, f'{correct}_hidden.png'))


        #########

        prob_array = np.zeros((num_tokens,num_layers))

        for num_layers, pos in product(range(num_layers), range(num_tokens)):
            prob_array[pos, num_layers] = exp['mlp'][str(num_layers)][str(pos)]

        plt.figure(figsize=(10,6))
        plt.xticks(np.arange(0,num_layers,5)+0.5, np.arange(0,num_layers,5))
        plt.yticks()
        plt.yticks(np.arange(0,num_tokens)+0.5)
        im = plt.pcolormesh(prob_array, cmap="Greens")
        plt.gca().invert_yaxis()
        cbar = plt.colorbar(im)
        cbar.ax.set_title(f"p({correct})", y=-0.07)
        plt.savefig(os.path.join(model_dir, f'{correct}_mlp.png'))

        #########

        prob_array = np.zeros((num_tokens,num_layers))

        for num_layers, pos in product(range(num_layers), range(num_tokens)):
            prob_array[pos, num_layers] = exp['attn'][str(num_layers)][str(pos)]

        plt.figure(figsize=(10,6))
        plt.xticks(np.arange(0,num_layers,5)+0.5, np.arange(0,num_layers,5))
        plt.yticks()
        plt.yticks(np.arange(0,num_tokens)+0.5)
        im = plt.pcolormesh(prob_array, cmap="Reds")
        plt.gca().invert_yaxis()
        cbar = plt.colorbar(im)
        cbar.ax.set_title(f"p({correct})", y=-0.07)
        plt.savefig(os.path.join(model_dir, f'{correct}_attn.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['gpt2', 'gpt-neo'], default='gpt2')
    parser.add_argument('--model_size', default='large')
    parser.add_argument('--results_dir', default='./rome_results')
    
    args = parser.parse_args()
    
    main(args)