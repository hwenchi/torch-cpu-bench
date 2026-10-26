from transformers import CLIPVisionConfig, CLIPVisionModel

from .inputs import IMAGENET_SIZE, image

HEAD_DIM = 64


def clip_image_encoder(layers, hidden, patch):
    # CLIP's image tower: the part that runs per image when embedding photos for search or zero-shot labels.
    config = CLIPVisionConfig(
        num_hidden_layers=layers, hidden_size=hidden, num_attention_heads=hidden // HEAD_DIM,
        intermediate_size=hidden * 4, patch_size=patch, image_size=IMAGENET_SIZE,
    )
    return lambda: CLIPVisionModel(config)


imagenet = image(IMAGENET_SIZE)

MODELS = [
    {"name": "clip_vit_b_32", "family": "CLIP", "task": "Image-text embedding", "build": clip_image_encoder(layers=12, hidden=768, patch=32), "input": imagenet},
    {"name": "clip_vit_b_16", "family": "CLIP", "task": "Image-text embedding", "build": clip_image_encoder(layers=12, hidden=768, patch=16), "input": imagenet},
    {"name": "clip_vit_l_14", "family": "CLIP", "task": "Image-text embedding", "build": clip_image_encoder(layers=24, hidden=1024, patch=14), "input": imagenet},
]
