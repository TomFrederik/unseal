import argparse
import json
import logging
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.file_utils import RepositoryNotFoundError

from info_flow import eval_model
import hooks
import transformers_util as tutil

def main(args):
    if args.device is None: # use cuda except for large models --> #TODO check automatically if model fits on gpu
        device = 'cuda' if (torch.cuda.is_available() and args.model_size not in ['xl', '2.7B']) else 'cpu' # larger doesn't fit on my gpu
    else:
        device = args.device

    # assemble model name
    model_name = args.model
    if args.model_size is not None:
        model_name += '-' + args.model_size
    else:
        args.model_size = 'None' # for path creation/saving

    # Trying to load model, tokenizer and config
    model, tokenizer, config = tutil.load(model_name, args.model_dir)
    model.to(device)
    model.eval()
    # wrap model for hooking access
    model = hooks.HookedModel(model)

    # get number of layers
    num_layers = config.num_hidden_layers
    logging.info(f'{num_layers = }')

    # make directories
    os.makedirs(f'./info_results/{args.model}/{args.model_size}', exist_ok=True)

    # load data
    with open(args.text_file, "r") as f:
        data = json.load(f)
    prompts = data['prompts']
    entities = data['entity']
    
    # run experiments
    all_results = dict()
    for i in range(len(prompts)):
        base_text = prompts[i]
        entity = entities[i]

        results = eval_model(model, tokenizer, base_text, entity, num_layers, args.embedding_name, args.mlp_name, args.attn_name, args.trafo_name)
        results['prompt'] = base_text
        results['entity'] = entity
        all_results[i] = results

        # save results after every sub-experiment
        with open(f'./info_results/{args.model}/{args.model_size}/results.json', 'w') as f:
            json.dump(all_results, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # model args
    parser.add_argument('--model', default='gpt2')
    parser.add_argument('--model_size', type=str, help='Model size, e.g. large or xl for gpt2 or 125M for gpt-neo', default=None)
    parser.add_argument('--model_dir', default=None, help='HF directory in which to look for the specified model, e.g. EleutherAI')
    
    # these args need to be changed from default when using non-GPT based models
    parser.add_argument('--trafo_name', default='h', help='name of the transformer layers in the model')
    parser.add_argument('--embedding_name', default='wte', help='name of the embedding layer in the model')
    parser.add_argument('--mlp_name', default='mlp', help='name of the MLP layers in the model')
    parser.add_argument('--attn_name', default='attn', help='name of the Attentino layers in the model')

    # Misc
    parser.add_argument('--text_file', default='prompts.json', help='File that contains the data')
    parser.add_argument('--device', default=None)


    args = parser.parse_args()
    
    main(args)