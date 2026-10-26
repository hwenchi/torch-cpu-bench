import torch
import torchvision.models as tvm
import torchvision.models.detection as tvd
import torchvision.models.segmentation as tvs
from transformers import BertConfig, BertModel, GPT2Config, GPT2Model, LlamaConfig, LlamaModel
from transformers import Wav2Vec2Config, Wav2Vec2Model, WhisperConfig
from transformers.models.whisper.modeling_whisper import WhisperEncoder


def image(size, width=None):
    return lambda: torch.randn(1, 3, size, width or size)


def tokens(length, vocab=30522):
    return lambda: torch.randint(0, vocab, (1, length))


def waveform(seconds):
    return lambda: torch.randn(1, 16000 * seconds)


def mel(bins=80, frames=3000):
    return lambda: torch.randn(1, bins, frames)


class Batched(torch.nn.Module):
    # torchvision detectors take a list of images; this lets them take a batch tensor like everything else.
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(list(x))


def bert(layers, hidden):
    config = BertConfig(
        num_hidden_layers=layers,
        hidden_size=hidden,
        num_attention_heads=hidden // 64,
        intermediate_size=hidden * 4,
    )
    return lambda: BertModel(config, add_pooling_layer=False)


def gpt2(layers, hidden):
    config = GPT2Config(n_layer=layers, n_embd=hidden, n_head=hidden // 64)
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


def whisper_encoder(layers, hidden):
    config = WhisperConfig(
        encoder_layers=layers,
        d_model=hidden,
        encoder_attention_heads=hidden // 64,
        encoder_ffn_dim=hidden * 4,
    )
    return lambda: WhisperEncoder(config)


def wav2vec2(layers, hidden, large=False):
    config = Wav2Vec2Config(
        num_hidden_layers=layers,
        hidden_size=hidden,
        num_attention_heads=hidden // 64,
        intermediate_size=hidden * 4,
        feat_extract_norm="layer" if large else "group",
        do_stable_layer_norm=large,
    )

    def build():
        model = Wav2Vec2Model(config)
        # FlopCounterMode can't see through the weight_norm parametrization; bake it into a plain weight.
        torch.nn.utils.parametrize.remove_parametrizations(model.encoder.pos_conv_embed.conv, "weight")
        return model

    return build


SUITE = [
    {"name": "resnet18", "family": "ResNet", "domain": "vision", "build": lambda: tvm.resnet18(), "input": image(224)},
    {"name": "resnet34", "family": "ResNet", "domain": "vision", "build": lambda: tvm.resnet34(), "input": image(224)},
    {"name": "resnet50", "family": "ResNet", "domain": "vision", "build": lambda: tvm.resnet50(), "input": image(224)},
    {"name": "resnet101", "family": "ResNet", "domain": "vision", "build": lambda: tvm.resnet101(), "input": image(224)},

    {"name": "efficientnet_b0", "family": "EfficientNet", "domain": "vision", "build": lambda: tvm.efficientnet_b0(), "input": image(224)},
    {"name": "efficientnet_b2", "family": "EfficientNet", "domain": "vision", "build": lambda: tvm.efficientnet_b2(), "input": image(260)},
    {"name": "efficientnet_b4", "family": "EfficientNet", "domain": "vision", "build": lambda: tvm.efficientnet_b4(), "input": image(380)},

    {"name": "mobilenet_v3_small", "family": "MobileNetV3", "domain": "vision", "build": lambda: tvm.mobilenet_v3_small(), "input": image(224)},
    {"name": "mobilenet_v3_large", "family": "MobileNetV3", "domain": "vision", "build": lambda: tvm.mobilenet_v3_large(), "input": image(224)},

    {"name": "convnext_tiny", "family": "ConvNeXt", "domain": "vision", "build": lambda: tvm.convnext_tiny(), "input": image(224)},
    {"name": "convnext_small", "family": "ConvNeXt", "domain": "vision", "build": lambda: tvm.convnext_small(), "input": image(224)},
    {"name": "convnext_base", "family": "ConvNeXt", "domain": "vision", "build": lambda: tvm.convnext_base(), "input": image(224)},

    {"name": "vit_b_32", "family": "ViT", "domain": "vision", "build": lambda: tvm.vit_b_32(), "input": image(224)},
    {"name": "vit_b_16", "family": "ViT", "domain": "vision", "build": lambda: tvm.vit_b_16(), "input": image(224)},
    {"name": "vit_l_16", "family": "ViT", "domain": "vision", "build": lambda: tvm.vit_l_16(), "input": image(224)},

    {"name": "swin_t", "family": "Swin", "domain": "vision", "build": lambda: tvm.swin_t(), "input": image(224)},
    {"name": "swin_s", "family": "Swin", "domain": "vision", "build": lambda: tvm.swin_s(), "input": image(224)},
    {"name": "swin_b", "family": "Swin", "domain": "vision", "build": lambda: tvm.swin_b(), "input": image(224)},

    {"name": "fasterrcnn_mobilenet_v3_320", "family": "Faster R-CNN", "domain": "detection",
     "build": lambda: Batched(tvd.fasterrcnn_mobilenet_v3_large_320_fpn(weights_backbone=None)), "input": image(480, 640)},
    {"name": "fasterrcnn_mobilenet_v3", "family": "Faster R-CNN", "domain": "detection",
     "build": lambda: Batched(tvd.fasterrcnn_mobilenet_v3_large_fpn(weights_backbone=None)), "input": image(480, 640)},
    {"name": "fasterrcnn_resnet50", "family": "Faster R-CNN", "domain": "detection",
     "build": lambda: Batched(tvd.fasterrcnn_resnet50_fpn(weights_backbone=None)), "input": image(480, 640)},

    {"name": "deeplabv3_mobilenet_v3", "family": "DeepLabV3", "domain": "segmentation",
     "build": lambda: tvs.deeplabv3_mobilenet_v3_large(weights_backbone=None), "input": image(520)},
    {"name": "deeplabv3_resnet50", "family": "DeepLabV3", "domain": "segmentation",
     "build": lambda: tvs.deeplabv3_resnet50(weights_backbone=None), "input": image(520)},
    {"name": "deeplabv3_resnet101", "family": "DeepLabV3", "domain": "segmentation",
     "build": lambda: tvs.deeplabv3_resnet101(weights_backbone=None), "input": image(520)},

    {"name": "bert_mini", "family": "BERT", "domain": "text", "build": bert(4, 256), "input": tokens(128)},
    {"name": "bert_small", "family": "BERT", "domain": "text", "build": bert(4, 512), "input": tokens(128)},
    {"name": "bert_medium", "family": "BERT", "domain": "text", "build": bert(8, 512), "input": tokens(128)},
    {"name": "bert_base", "family": "BERT", "domain": "text", "build": bert(12, 768), "input": tokens(128)},

    {"name": "gpt2", "family": "GPT-2", "domain": "text", "build": gpt2(12, 768), "input": tokens(128, 50257)},
    {"name": "gpt2_medium", "family": "GPT-2", "domain": "text", "build": gpt2(24, 1024), "input": tokens(128, 50257)},
    {"name": "gpt2_large", "family": "GPT-2", "domain": "text", "build": gpt2(36, 1280), "input": tokens(128, 50257)},

    {"name": "smollm_135m", "family": "Llama", "domain": "text", "build": llama(30, 576, 9, 3, 1536, 49152), "input": tokens(128, 49152)},
    {"name": "smollm_360m", "family": "Llama", "domain": "text", "build": llama(32, 960, 15, 5, 2560, 49152), "input": tokens(128, 49152)},
    {"name": "llama3_2_1b", "family": "Llama", "domain": "text", "build": llama(16, 2048, 32, 8, 8192, 128256), "input": tokens(128, 128256)},

    {"name": "whisper_tiny_encoder", "family": "Whisper", "domain": "audio", "build": whisper_encoder(4, 384), "input": mel()},
    {"name": "whisper_base_encoder", "family": "Whisper", "domain": "audio", "build": whisper_encoder(6, 512), "input": mel()},
    {"name": "whisper_small_encoder", "family": "Whisper", "domain": "audio", "build": whisper_encoder(12, 768), "input": mel()},

    {"name": "wav2vec2_base", "family": "Wav2Vec2", "domain": "audio", "build": wav2vec2(12, 768), "input": waveform(1)},
    {"name": "wav2vec2_large", "family": "Wav2Vec2", "domain": "audio", "build": wav2vec2(24, 1024, large=True), "input": waveform(1)},
]
