import json

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from hooks import FullModelHooks, hidden_patch_hook_fn, additive_noise, mlp_patch_interval, attn_patch_interval



model_size = 'xl'
device = 'cuda' if (torch.cuda.is_available() and model_size != 'xl') else 'cpu' # xl doesn't fit on my gpu

size2suffix = {'small':'', 'medium':'-medium', 'large':'-large', 'xl':'-xl'}
num_heads = {'small':12, 'medium':24, 'large':36, 'xl':48}[model_size]

tokenizer = GPT2Tokenizer.from_pretrained(f"gpt2{size2suffix[model_size]}")
model = GPT2LMHeadModel.from_pretrained(f"gpt2{size2suffix[model_size]}")
model.to(device)
model.eval()


hooks = FullModelHooks(model)
print(hooks)

base_text = "The Big Bang Theory premieres on"
encoded_input = tokenizer(base_text, return_tensors='pt').to(device)
num_tokens = encoded_input['input_ids'].shape[1]
print(f"{encoded_input['input_ids'].shape = }") 


encoded_output = tokenizer(" CBS", return_tensors='pt').to(device)
correct_id = encoded_output['input_ids'][0,0]
# print(f'{correct_id = }')

output = model(**encoded_input)
# probs = torch.softmax(output['logits'][0,-1,:], 0)
# print(f"{probs[correct_id] = }")
# print(f"{probs.max() = }")
# print(f"{probs.argmax() = }")
# max_token = tokenizer.batch_decode(probs.argmax()[None])
# print(f'{max_token = }')

# save outputs of model on correct text -> very expensive computation, maybe iteratively save to disk instead?
correct_hidden = {key: hook.features[0].to('cpu') for key, hook in hooks.items() if key.startswith('transformer->h->')}

hooks.add_custom_hook('transformer->wte', 'embedding_noise', additive_noise(indices=":4,:", std=0.1))

results = dict(mlp=dict(), attn=dict(), hidden=dict())

# mlp
for center in range(num_heads):
    results['mlp'][center] = dict()
    for pos in range(num_tokens):
        new_keys = []
        for layer in range(max(0,center-5), min(num_heads, center+5)): #TODO currently removing and recreating hooks too often -> make as sliding window
            old_key = f'transformer->h->{layer}->mlp'
            new_key = f'patch_h{layer}_mlp_pos{pos}'
            new_keys.append(new_key)
            hooks.add_custom_hook(old_key, new_key, hidden_patch_hook_fn(pos, correct_hidden[old_key][pos]))
        output = model(**encoded_input)

        prob = torch.softmax(output["logits"][0,-1,:], 0)[correct_id].item()
        results['mlp'][center][pos] = prob
        
        for key in new_keys:
            hooks.remove_hook(key)

# attn
for center in range(num_heads):
    results['attn'][center] = dict()
    for pos in range(num_tokens):
        new_keys = []
        for layer in range(max(0,center-5), min(num_heads, center+5)): #TODO currently removing and recreating hooks too often -> make as sliding window
            old_key = f'transformer->h->{layer}->attn'
            new_key = f'patch_h{layer}_attn_pos{pos}'
            new_keys.append(new_key)
            hooks.add_custom_hook(old_key, new_key, hidden_patch_hook_fn(pos, correct_hidden[old_key][0][pos]))
        output = model(**encoded_input)

        prob = torch.softmax(output["logits"][0,-1,:], 0)[correct_id].item()
        results['attn'][center][pos] = prob
        
        for key in new_keys:
            hooks.remove_hook(key)

# hidden
for head_num in range(num_heads):
    results['hidden'][head_num] = dict()
    for pos in range(num_tokens):
        old_key = f'transformer->h->{head_num}'
        new_key = f'patch_h{head_num}_pos{pos}'
        hooks.add_custom_hook(old_key, new_key, hidden_patch_hook_fn(pos, correct_hidden[old_key][0][pos]))

        output = model(**encoded_input)

        prob = torch.softmax(output["logits"][0,-1,:], 0)[correct_id].item()
        results['hidden'][head_num][pos] = prob

        hooks.remove_hook(new_key)

# print(results)
with open(f'./rome_results_{model_size}.json', 'w') as f:
    json.dump(results, f)

