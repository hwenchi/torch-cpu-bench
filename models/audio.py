import torch
from transformers import ASTConfig, ASTModel, EncodecConfig, EncodecModel, Wav2Vec2Config, Wav2Vec2Model, WhisperConfig
from transformers.models.whisper.modeling_whisper import WhisperEncoder

from .inputs import AST_FRAMES, AST_MELS, CODEC_SAMPLE_RATE, WHISPER_FRAMES, WHISPER_MELS, features, waveform

HEAD_DIM = 64


def whisper_encoder(layers, hidden):
    config = WhisperConfig(
        encoder_layers=layers,
        d_model=hidden,
        encoder_attention_heads=hidden // HEAD_DIM,
        encoder_ffn_dim=hidden * 4,
        num_mel_bins=WHISPER_MELS,
    )
    return lambda: WhisperEncoder(config)


def wav2vec2(layers, hidden, large=False):
    config = Wav2Vec2Config(
        num_hidden_layers=layers,
        hidden_size=hidden,
        num_attention_heads=hidden // HEAD_DIM,
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


def encodec():
    def build():
        model = EncodecModel(EncodecConfig())
        # Every conv uses weight_norm, which FlopCounterMode can't see through; bake each into a plain weight.
        for module in model.modules():
            if torch.nn.utils.parametrize.is_parametrized(module, "weight"):
                torch.nn.utils.parametrize.remove_parametrizations(module, "weight")
        return model

    return build


whisper_input = features(WHISPER_MELS, WHISPER_FRAMES, label="30 s of audio as a spectrogram")
one_second = waveform(1)

MODELS = [
    {"name": "whisper_tiny_encoder", "family": "Whisper", "task": "Speech recognition", "build": whisper_encoder(layers=4, hidden=384), "input": whisper_input},
    {"name": "whisper_base_encoder", "family": "Whisper", "task": "Speech recognition", "build": whisper_encoder(layers=6, hidden=512), "input": whisper_input},
    {"name": "whisper_small_encoder", "family": "Whisper", "task": "Speech recognition", "build": whisper_encoder(layers=12, hidden=768), "input": whisper_input},
    {"name": "wav2vec2_base", "family": "Wav2Vec2", "task": "Speech recognition", "build": wav2vec2(layers=12, hidden=768), "input": one_second},
    {"name": "wav2vec2_large", "family": "Wav2Vec2", "task": "Speech recognition", "build": wav2vec2(layers=24, hidden=1024, large=True), "input": one_second},
    {"name": "ast_base", "family": "AST", "task": "Audio classification",
     "build": lambda: ASTModel(ASTConfig(max_length=AST_FRAMES, num_mel_bins=AST_MELS)), "input": features(AST_FRAMES, AST_MELS, label="10 s of audio as a spectrogram")},
    {"name": "encodec_24khz", "family": "EnCodec", "task": "Audio codec",
     "build": encodec(), "input": waveform(1, rate=CODEC_SAMPLE_RATE, channels=1)},
]
