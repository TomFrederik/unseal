import json

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from hooks import FullModelHooks, hidden_patch_hook_fn


# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

from transformers import GPT2Tokenizer, GPT2LMHeadModel

model_size = 'xl'
size2suffix = {'small':'', 'medium':'-medium', 'large':'-large', 'xl':'-xl'}
num_heads = {'small':12, 'medium':24, 'large':36, 'xl':48}[model_size]

tokenizer = GPT2Tokenizer.from_pretrained(f"gpt2{size2suffix[model_size]}")
model = GPT2LMHeadModel.from_pretrained(f"gpt2{size2suffix[model_size]}")
model.to(device)
model.eval()


hooks = FullModelHooks(model)
print(hooks)

base_text = "The Big Bang Theory airs on"
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
correct_hidden = {key: hook.features[0][0] for key, hook in hooks.items() if key.startswith('transformer->h->')}

# add noise hook
def noise_func(output):
    noise = 0.1 * torch.randn_like(output[:,:4]) # TODO fix hardcoding of subject position
    output[:,:4] += noise
    return output

hooks.add_custom_hook('transformer->wte', 'embedding_noise', noise_func)

results = dict()

for head_num in range(num_heads):
    results[head_num] = dict()
    for pos in range(num_tokens):
        old_key = f'transformer->h->{head_num}'
        new_key = f'patch_h{head_num}_pos{pos}'
        hooks.add_custom_hook(old_key, new_key, hidden_patch_hook_fn(pos, correct_hidden[old_key][pos]))

        output = model(**encoded_input)

        prob = torch.softmax(output["logits"][0,-1,:], 0)[correct_id].item()
        results[head_num][pos] = prob

        hooks.remove_hook(new_key)

# print(results)
with open(f'./rome_results_{model_size}.json', 'w') as f:
    json.dump(results, f)

