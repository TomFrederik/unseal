import torch

from hooks import hidden_patch_hook_fn, additive_noise

def eval_model(model, tokenizer, base_text: str, entity_indices: str, correct_output_text: str, num_layers: int, hooks):
    device = next(model.parameters()).device

    encoded_base_text = tokenizer(base_text, return_tensors='pt').to(device)
    num_tokens = encoded_base_text['input_ids'].shape[1]
    
    encoded_correct_output_text = tokenizer(correct_output_text, return_tensors='pt').to(device)
    correct_id = encoded_correct_output_text['input_ids'].item()

    # initial pass
    output = model(**encoded_base_text)

    # save outputs of model on correct text
    correct_hidden = {key: hook.features[0].to('cpu') for key, hook in hooks.items() if key.startswith('transformer->h->')}

    # add embedding noise
    hooks.add_custom_hook('transformer->wte', 'embedding_noise', additive_noise(indices=f"{entity_indices},:", std=0.1))

    # init results
    results = dict(mlp=dict(), attn=dict(), hidden=dict())

    # mlp
    print('Patching MLPs..')
    for center in range(num_layers):
        results['mlp'][center] = dict()
        for pos in range(num_tokens):
            new_keys = []
            for layer in range(max(0,center-5), min(num_layers, center+5)): #TODO currently removing and recreating hooks too often -> make as sliding window
                old_key = f'transformer->h->{layer}->mlp'
                new_key = f'patch_h{layer}_mlp_pos{pos}'
                new_keys.append(new_key)
                hooks.add_custom_hook(old_key, new_key, hidden_patch_hook_fn(pos, correct_hidden[old_key][pos]))
            output = model(**encoded_base_text)

            prob = torch.softmax(output["logits"][0,-1,:], 0)[correct_id].item()
            results['mlp'][center][pos] = prob
            
            for key in new_keys:
                hooks.remove_hook(key)

    # attn
    print('Patching attentions...')
    for center in range(num_layers):
        results['attn'][center] = dict()
        for pos in range(num_tokens):
            new_keys = []
            for layer in range(max(0,center-5), min(num_layers, center+5)): #TODO currently removing and recreating hooks too often -> make as sliding window
                old_key = f'transformer->h->{layer}->attn'
                new_key = f'patch_h{layer}_attn_pos{pos}'
                new_keys.append(new_key)
                hooks.add_custom_hook(old_key, new_key, hidden_patch_hook_fn(pos, correct_hidden[old_key][0][pos]))
            output = model(**encoded_base_text)

            prob = torch.softmax(output["logits"][0,-1,:], 0)[correct_id].item()
            results['attn'][center][pos] = prob
            
            for key in new_keys:
                hooks.remove_hook(key)

    # hidden
    print('Patching hidden states...')
    for layer_num in range(num_layers):
        results['hidden'][layer_num] = dict()
        for pos in range(num_tokens):
            old_key = f'transformer->h->{layer_num}'
            new_key = f'patch_h{layer_num}_pos{pos}'
            hooks.add_custom_hook(old_key, new_key, hidden_patch_hook_fn(pos, correct_hidden[old_key][0][pos]))

            output = model(**encoded_base_text)

            prob = torch.softmax(output["logits"][0,-1,:], 0)[correct_id].item()
            results['hidden'][layer_num][pos] = prob

            hooks.remove_hook(new_key)
    
    hooks.remove_hook('embedding_noise')
    
    return results

