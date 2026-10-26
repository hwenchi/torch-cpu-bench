import torchvision.models.video as tvv
from transformers import VideoMAEConfig, VideoMAEModel

from .inputs import CLIP_FRAMES, IMAGENET_SIZE, SMALL_CLIP_SIZE, clip, frames_first

small_clip = clip(CLIP_FRAMES, SMALL_CLIP_SIZE)
full_clip = clip(CLIP_FRAMES, IMAGENET_SIZE)

MODELS = [
    {"name": "r3d_18", "family": "ResNet 3D", "task": "Action recognition", "build": lambda: tvv.r3d_18(), "input": small_clip},
    {"name": "mc3_18", "family": "ResNet 3D", "task": "Action recognition", "build": lambda: tvv.mc3_18(), "input": small_clip},
    {"name": "r2plus1d_18", "family": "ResNet 3D", "task": "Action recognition", "build": lambda: tvv.r2plus1d_18(), "input": small_clip},
    {"name": "s3d", "family": "S3D", "task": "Action recognition", "build": lambda: tvv.s3d(), "input": full_clip},
    {"name": "mvit_v1_b", "family": "MViT", "task": "Action recognition", "build": lambda: tvv.mvit_v1_b(), "input": full_clip},
    {"name": "swin3d_t", "family": "Video Swin", "task": "Action recognition", "build": lambda: tvv.swin3d_t(), "input": full_clip},
    {"name": "videomae_base", "family": "VideoMAE", "task": "Action recognition",
     "build": lambda: VideoMAEModel(VideoMAEConfig()), "input": frames_first(CLIP_FRAMES, IMAGENET_SIZE)},
]
