from transformers import (
    BertConfig, BertModel, DistilBertConfig, DistilBertModel, GPT2Config, GPT2Model, LlamaConfig, LlamaModel,
    MambaConfig, MambaModel, ModernBertConfig, ModernBertModel, T5Config, T5Model,
)

from .inputs import (
    BERT_VOCAB, GPT2_VOCAB, LLAMA3_VOCAB, MAMBA_VOCAB, MODERNBERT_VOCAB, SEQ_LEN, SMOLLM_VOCAB, T5_VOCAB, described, tokens,
)
from .wrappers import FirstDecoderStep

HEAD_DIM = 64


def bert(layers, hidden, heads=None):
    config = BertConfig(
        num_hidden_layers=layers,
        hidden_size=hidden,
        num_attention_heads=heads or hidden // HEAD_DIM,
        intermediate_size=hidden * 4,
    )
    return lambda: BertModel(config, add_pooling_layer=False)


def gpt2(layers, hidden):
    config = GPT2Config(n_layer=layers, n_embd=hidden, n_head=hidden // HEAD_DIM)
    return lambda: GPT2Model(config)


def llama(layers, hidden, heads, kv_heads, intermediate, vocab):
    config = LlamaConfig(
        num_hidden_layers=layers,
        hidden_size=hidden,
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        intermediate_size=intermediate,
        vocab_size=vocab,
    )
    return lambda: LlamaModel(config)


def t5(layers, hidden, heads, ff):
    config = T5Config(num_layers=layers, d_model=hidden, num_heads=heads, d_ff=ff, d_kv=hidden // heads)
    return lambda: FirstDecoderStep(T5Model(config))


def mamba(layers, hidden):
    config = MambaConfig(num_hidden_layers=layers, hidden_size=hidden, vocab_size=MAMBA_VOCAB)
    return lambda: MambaModel(config)


bert_tokens = tokens(SEQ_LEN, BERT_VOCAB)
t5_prompt = described(tokens(SEQ_LEN, T5_VOCAB), f"{SEQ_LEN}-token prompt, then 1 output token")

MODELS = [
    # Encoders
    {"name": "minilm_l6", "family": "Sentence embedding", "task": "Text embedding", "build": bert(layers=6, hidden=384, heads=12), "input": bert_tokens},
    {"name": "bge_small", "family": "Sentence embedding", "task": "Text embedding", "build": bert(layers=12, hidden=384, heads=12), "input": bert_tokens},
    {"name": "bge_large", "family": "Sentence embedding", "task": "Text embedding", "build": bert(layers=24, hidden=1024, heads=16), "input": bert_tokens},
    {"name": "bert_mini", "family": "BERT", "task": "Text understanding", "build": bert(layers=4, hidden=256), "input": bert_tokens},
    {"name": "bert_small", "family": "BERT", "task": "Text understanding", "build": bert(layers=4, hidden=512), "input": bert_tokens},
    {"name": "bert_medium", "family": "BERT", "task": "Text understanding", "build": bert(layers=8, hidden=512), "input": bert_tokens},
    {"name": "bert_base", "family": "BERT", "task": "Text understanding", "build": bert(layers=12, hidden=768), "input": bert_tokens},
    {"name": "distilbert", "family": "DistilBERT", "task": "Text understanding",
     "build": lambda: DistilBertModel(DistilBertConfig()), "input": bert_tokens},
    {"name": "modernbert_base", "family": "ModernBERT", "task": "Text understanding",
     "build": lambda: ModernBertModel(ModernBertConfig()), "input": tokens(SEQ_LEN, MODERNBERT_VOCAB)},

    # Encoder-decoder
    {"name": "t5_small", "family": "T5", "task": "Text-to-text", "build": t5(layers=6, hidden=512, heads=8, ff=2048), "input": t5_prompt},
    {"name": "t5_base", "family": "T5", "task": "Text-to-text", "build": t5(layers=12, hidden=768, heads=12, ff=3072), "input": t5_prompt},

    # Decoders
    {"name": "gpt2", "family": "GPT-2", "task": "Text generation", "build": gpt2(layers=12, hidden=768), "input": tokens(SEQ_LEN, GPT2_VOCAB)},
    {"name": "gpt2_medium", "family": "GPT-2", "task": "Text generation", "build": gpt2(layers=24, hidden=1024), "input": tokens(SEQ_LEN, GPT2_VOCAB)},
    {"name": "gpt2_large", "family": "GPT-2", "task": "Text generation", "build": gpt2(layers=36, hidden=1280), "input": tokens(SEQ_LEN, GPT2_VOCAB)},
    {"name": "smollm_135m", "family": "Llama", "task": "Text generation",
     "build": llama(layers=30, hidden=576, heads=9, kv_heads=3, intermediate=1536, vocab=SMOLLM_VOCAB), "input": tokens(SEQ_LEN, SMOLLM_VOCAB)},
    {"name": "smollm_360m", "family": "Llama", "task": "Text generation",
     "build": llama(layers=32, hidden=960, heads=15, kv_heads=5, intermediate=2560, vocab=SMOLLM_VOCAB), "input": tokens(SEQ_LEN, SMOLLM_VOCAB)},
    {"name": "llama3_2_1b", "family": "Llama", "task": "Text generation",
     "build": llama(layers=16, hidden=2048, heads=32, kv_heads=8, intermediate=8192, vocab=LLAMA3_VOCAB), "input": tokens(SEQ_LEN, LLAMA3_VOCAB)},
    {"name": "mamba_130m", "family": "Mamba", "task": "Text generation", "build": mamba(layers=24, hidden=768), "input": tokens(SEQ_LEN, MAMBA_VOCAB)},
]
