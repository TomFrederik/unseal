import torch

import hooks
import hooks.common_hooks as common_hooks

def eval_model(
    model, 
    tokenizer, 
    base_text: str, 
    entity_indices: str, 
    num_layers: int,
    embedding_name: str = 'wte',
    mlp_name: str = 'mlp',
    attn_name: str = 'attn',
    trafo_name: str = 'h',
):
    device = next(model.parameters()).device
    
    encoded_base_text = tokenizer(base_text, return_tensors='pt').to(device)
    
    # split correct target token from sentence
    correct_id = encoded_base_text['input_ids'][0,-1].item()
    encoded_base_text['input_ids'] = encoded_base_text['input_ids'][:,:-1]
    encoded_base_text['attention_mask'] = encoded_base_text['attention_mask'][:,:-1]
    num_tokens = encoded_base_text['input_ids'].shape[1]
    output_hooks_names = [f'transformer->{trafo_name}->{layer}{suffix}' for layer in range(num_layers) for suffix in ['', f'->{mlp_name}', f'->{attn_name}']]

    output_hooks = [hooks.Hook(name, common_hooks.save_output, name) for name in output_hooks_names]
    model(**encoded_base_text, hooks=output_hooks)

    # add embedding noise
    model_hooks = [
        hooks.Hook(
            layer_name=f'transformer->{embedding_name}', 
            func=hooks.rome_hooks.additive_noise(indices=f"{entity_indices},:", std=0.1),
            key='embedding_noise',
        )
    ]

    # init results
    results = dict(mlp=dict(), attn=dict(), hidden=dict())

    # # mlp
    print('Patching MLPs..')
    for center in range(num_layers):
        results['mlp'][center] = dict()
        for pos in range(num_tokens):
            new_hooks = []
            for layer in range(max(0,center-5), min(num_layers, center+5)): #TODO currently removing and recreating hooks too often -> make as sliding window?
                layer_name = f'transformer->{trafo_name}->{layer}->{mlp_name}'
                hook_key = f'patch_{trafo_name}{layer}_{mlp_name}_pos{pos}'
                new_hooks.append(hooks.Hook(
                    layer_name=layer_name,
                    func=hooks.rome_hooks.hidden_patch_hook_fn(pos, model.save_ctx[layer_name]['output'][0,pos]),
                    key=hook_key,
                ))

            output = model(**encoded_base_text, hooks=model_hooks + new_hooks)
            prob = torch.softmax(output["logits"][0,-1,:], 0)[correct_id].item()
            results['mlp'][center][pos] = prob
            

    # # attn
    print('Patching attentions...')
    for center in range(num_layers):
        results['attn'][center] = dict()
        for pos in range(num_tokens):
            new_hooks = []
            for layer in range(max(0,center-5), min(num_layers, center+5)): #TODO currently removing and recreating hooks too often -> make as sliding window?
                layer_name = f'transformer->{trafo_name}->{layer}->{attn_name}'
                hook_key = f'patch_{trafo_name}{layer}_{attn_name}_pos{pos}'
                new_hooks.append(hooks.Hook(
                    layer_name=layer_name,
                    func=hooks.rome_hooks.hidden_patch_hook_fn(pos, model.save_ctx[layer_name]['output'][0][0,pos]),
                    key=hook_key,
                ))

            output = model(**encoded_base_text, hooks=model_hooks + new_hooks)
            prob = torch.softmax(output["logits"][0,-1,:], 0)[correct_id].item()
            results['attn'][center][pos] = prob

    # hidden
    print('Patching hidden states...')
    for layer_num in range(num_layers):
        results['hidden'][layer_num] = dict()
        for pos in range(num_tokens):
            layer_name = f'transformer->{trafo_name}->{layer_num}'
            hook_key = f'patch_{trafo_name}{layer_num}_pos{pos}'
            new_hook = hooks.Hook(
                layer_name=layer_name,
                func=hooks.rome_hooks.hidden_patch_hook_fn(pos, model.save_ctx[layer_name]['output'][0][0,pos]),
                key=hook_key,
            )
            output = model(**encoded_base_text, hooks=model_hooks + [new_hook])

            prob = torch.softmax(output["logits"][0,-1,:], 0)[correct_id].item()
            results['hidden'][layer_num][pos] = prob

    return results

