import torch
import unseal.transformers_util as tutil
from unseal.hooks import HookedModel

def test_load_model():
    model, tokenizer, config = tutil.load_from_pretrained('gpt2')
    assert model is not None
    assert tokenizer is not None
    assert config is not None

def test_load_model_with_dir():
    model, tokenizer, config = tutil.load_from_pretrained('gpt-neo-125M', model_dir='EleutherAI')
    assert model is not None
    assert tokenizer is not None
    assert config is not None

def test_load_model_eleuther_without_dir():
    model, tokenizer, config = tutil.load_from_pretrained('gpt-neo-125M')
    assert model is not None
    assert tokenizer is not None
    assert config is not None

def test_load_model_with_low_mem():
    model, tokenizer, config = tutil.load_from_pretrained('gpt2', low_cpu_mem_usage=True)
    assert model is not None
    assert tokenizer is not None
    assert config is not None

def test_get_num_layers_gpt2():
    model, *_ = tutil.load_from_pretrained('gpt2')
    model = HookedModel(model)
    assert tutil.get_num_layers(model, 'transformer->h') == 12

def test_get_num_layers_transformer():
    model = torch.nn.Transformer(d_model=10, nhead=2, num_encoder_layers=0, num_decoder_layers=10)
    model = HookedModel(model)
    assert tutil.get_num_layers(model, 'decoder->layers')
    