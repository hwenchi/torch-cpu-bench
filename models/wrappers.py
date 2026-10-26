"""Adapters that give multi-input models a single-tensor forward, so the runner can treat every model the same."""
import torch
import torch.nn as nn

from .inputs import RGB


class Batched(nn.Module):
    # torchvision detectors take a list of images.
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(list(x))


class FramePair(nn.Module):
    # Optical flow takes two frames; the input stacks them on the channel axis.
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x[:, :RGB], x[:, RGB:])


class FirstDecoderStep(nn.Module):
    # Encoder-decoder: encode the prompt and run the decoder for one token.
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, ids):
        return self.model(input_ids=ids, decoder_input_ids=ids[:, :1])


class DenoiseStep(nn.Module):
    # One U-Net step of latent diffusion; a full image takes 20-50 of these.
    TIMESTEP = 500

    def __init__(self, unet, text_tokens, text_dim):
        super().__init__()
        self.unet = unet
        self.register_buffer("text", torch.randn(1, text_tokens, text_dim))

    def forward(self, latents):
        return self.unet(latents, self.TIMESTEP, encoder_hidden_states=self.text).sample


class VaeDecode(nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, latents):
        return self.vae.decode(latents).sample
