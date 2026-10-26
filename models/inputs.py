import torch

BATCH = 1

# Images
RGB = 3
GRAYSCALE = 1
IMAGENET_SIZE = 224
DETECTION_HEIGHT, DETECTION_WIDTH = 480, 640
SEGMENTATION_SIZE = 520
SAM_SIZE = 1024
DPT_SIZE = 384
FLOW_HEIGHT, FLOW_WIDTH = 368, 496
MEDICAL_SIZE = 256

# Video
CLIP_FRAMES = 16
SMALL_CLIP_SIZE = 112

# Text
SEQ_LEN = 128
BERT_VOCAB = 30522
GPT2_VOCAB = 50257
T5_VOCAB = 32128
MODERNBERT_VOCAB = 50368
SMOLLM_VOCAB = 49152
LLAMA3_VOCAB = 128256
MAMBA_VOCAB = 50280

# Audio
SAMPLE_RATE = 16000
CODEC_SAMPLE_RATE = 24000
WHISPER_MELS = 80
WHISPER_FRAMES = 3000  # 30 s of audio
AST_FRAMES = 1024
AST_MELS = 128

# Latent diffusion (Stable Diffusion 1.5 at 512 px)
LATENT_CHANNELS = 4
LATENT_SIZE = 64
TEXT_TOKENS = 77

# Structured data
GAN_NOISE = 100
ATARI_FRAMES = 4
ATARI_SIZE = 84
POINTS = 1024
SERIES_LEN = 512
SERIES_CHANNELS = 7
CORA_NODES = 2708
CORA_EDGES = 5278
CORA_FEATURES = 1433
CRITEO_DENSE = 13
CRITEO_SPARSE = 26


def image(height, width=None, channels=RGB):
    return lambda: torch.randn(BATCH, channels, height, width or height)


def clip(frames, size):
    # torchvision video models take (batch, channels, frames, height, width).
    return lambda: torch.randn(BATCH, RGB, frames, size, size)


def frames_first(frames, size):
    # VideoMAE takes (batch, frames, channels, height, width).
    return lambda: torch.randn(BATCH, frames, RGB, size, size)


def tokens(length, vocab):
    return lambda: torch.randint(0, vocab, (BATCH, length))


def waveform(seconds, rate=SAMPLE_RATE, channels=None):
    shape = (BATCH, rate * seconds) if channels is None else (BATCH, channels, rate * seconds)
    return lambda: torch.randn(*shape)


def features(*shape):
    return lambda: torch.randn(BATCH, *shape)


def click_log(rows):
    # Dense features and categorical ids packed into one float tensor; DLRM splits them back apart.
    return lambda: torch.cat([torch.randn(BATCH, CRITEO_DENSE), torch.randint(0, rows, (BATCH, CRITEO_SPARSE)).float()], dim=1)
