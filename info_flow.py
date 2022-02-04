import torch

import hooks
import hooks.common_hooks as common_hooks

def eval_model(model, tokenizer, base_text: str, entity_indices: str, correct_output_text: str, num_layers: int):
    device = next(model.parameters()).device

    encoded_base_text = tokenizer(base_text, return_tensors='pt').to(device)
    num_tokens = encoded_base_text['input_ids'].shape[1]
    
    encoded_correct_output_text = tokenizer(correct_output_text, return_tensors='pt').to(device)
    correct_id = encoded_correct_output_text['input_ids'].item()

    output_hooks_names = [f'transformer->h->{layer}{suffix}' for layer in range(num_layers) for suffix in ['', '->mlp', '->attn']]

    output_hooks = [hooks.Hook(name, common_hooks.save_output, name) for name in output_hooks_names]
    model(**encoded_base_text, hooks=output_hooks)

    # add embedding noise
    model_hooks = [
        hooks.Hook(
            layer_name='transformer->wte', 
            func=hooks.rome_hooks.additive_noise(indices=f"{entity_indices},:", std=0.1),
            key='embedding_noise',
        )
    ]

    # init results
    results = dict(mlp=dict(), attn=dict(), hidden=dict())

    # mlp
    print('Patching MLPs..')
    for center in range(num_layers):
        results['mlp'][center] = dict()
        for pos in range(num_tokens):
            new_hooks = []
            for layer in range(max(0,center-5), min(num_layers, center+5)): #TODO currently removing and recreating hooks too often -> make as sliding window?
                layer_name = f'transformer->h->{layer}->mlp'
                hook_key = f'patch_h{layer}_mlp_pos{pos}'
                new_hooks.append(hooks.Hook(
                    layer_name=layer_name,
                    func=hooks.rome_hooks.hidden_patch_hook_fn(pos, model.save_ctx[layer_name]['output'][0,pos]),
                    key=hook_key,
                ))

            output = model(**encoded_base_text, hooks=model_hooks + new_hooks)
            prob = torch.softmax(output["logits"][0,-1,:], 0)[correct_id].item()
            results['mlp'][center][pos] = prob
            

    # attn
    print('Patching attentions...')
    for center in range(num_layers):
        results['attn'][center] = dict()
        for pos in range(num_tokens):
            new_hooks = []
            for layer in range(max(0,center-5), min(num_layers, center+5)): #TODO currently removing and recreating hooks too often -> make as sliding window?
                layer_name = f'transformer->h->{layer}->attn'
                hook_key = f'patch_h{layer}_attn_pos{pos}'
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
            layer_name = f'transformer->h->{layer_num}'
            hook_key = f'patch_h{layer_num}_pos{pos}'
            new_hook = hooks.Hook(
                layer_name=layer_name,
                func=hooks.rome_hooks.hidden_patch_hook_fn(pos, model.save_ctx[layer_name]['output'][0][0,pos]),
                key=hook_key,
            )
            output = model(**encoded_base_text, hooks=model_hooks + [new_hook])

            prob = torch.softmax(output["logits"][0,-1,:], 0)[correct_id].item()
            results['hidden'][layer_num][pos] = prob

    return results

