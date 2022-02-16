import torch

from unseal import hooks
from transformers_util import load_from_pretrained, get_num_layers
from hooks.common_hooks import logit_hook

# function that generates the necessary data to generate the plots from the logits
# lense post, for a given sentence and a given model + tokenizer
def generate_logit_lense(model, tokenizer, sentence):
    tokenized_sentence = tokenizer.encode(sentence, return_tensors='pt').to('cpu')
    targets = tokenizer.encode(sentence)[1:]
    
    num_layers = get_num_layers(model)
    
    logit_hooks = [
        logit_hook(layer, model, target=targets) for layer in range(num_layers)
    ]
    
    model.forward(tokenized_sentence, hooks=logit_hooks)
    
    logits = [model.save_ctx[str(layer) + '_logits']['logits'][0] for layer in range(num_layers)]
    print(f"{logits[0].shape = }")
    print(f"{len(logits) = }")
    for l in logits:
        print(l[3,2]) # logit of predicting "2" at the appropriate position

model, tokenizer, config = load_from_pretrained('gpt2-xl')
model = hooks.HookedModel(model).to('cpu')

text = "After 1 comes 2. After 2 comes 3. After 3 comes 4."
generate_logit_lense(model, tokenizer, text)
