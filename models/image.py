import torchvision.models as tvm
import torchvision.models.detection as tvd
import torchvision.models.optical_flow as tvf
import torchvision.models.segmentation as tvs
from diffusers import AutoencoderKL, UNet2DConditionModel
from transformers import (
    DetrConfig, DetrForObjectDetection, Dinov2Config, Dinov2Model, DPTConfig, DPTForDepthEstimation, ResNetConfig,
    SamVisionConfig, SamVisionModel,
)

from .custom import DCGANGenerator, UNet
from .inputs import (
    DETECTION_HEIGHT, DETECTION_WIDTH, DPT_SIZE, FLOW_HEIGHT, FLOW_WIDTH, GAN_NOISE, GRAYSCALE, IMAGENET_SIZE,
    LATENT_CHANNELS, LATENT_SIZE, MEDICAL_SIZE, RGB, SAM_SIZE, SEGMENTATION_SIZE, TEXT_TOKENS, features, image,
)
from .wrappers import Batched, DenoiseStep, FramePair, VaeDecode


def dinov2(layers, hidden, heads):
    config = Dinov2Config(num_hidden_layers=layers, hidden_size=hidden, num_attention_heads=heads, image_size=IMAGENET_SIZE)
    return lambda: Dinov2Model(config)


def detr():
    config = DetrConfig(
        use_timm_backbone=False, use_pretrained_backbone=False,
        backbone_config=ResNetConfig(out_features=["stage4"]),
    )
    return lambda: DetrForObjectDetection(config)


def sd15_unet():
    def build():
        unet = UNet2DConditionModel(
            sample_size=LATENT_SIZE, block_out_channels=(320, 640, 1280, 1280),
            cross_attention_dim=768, attention_head_dim=8,
        )
        return DenoiseStep(unet, text_tokens=TEXT_TOKENS, text_dim=768)

    return build


def sd15_vae_decoder():
    return lambda: VaeDecode(AutoencoderKL(
        block_out_channels=(128, 256, 512, 512), latent_channels=LATENT_CHANNELS, layers_per_block=2,
        down_block_types=("DownEncoderBlock2D",) * 4, up_block_types=("UpDecoderBlock2D",) * 4,
    ))


imagenet = image(IMAGENET_SIZE)
detection = image(DETECTION_HEIGHT, DETECTION_WIDTH)
segmentation = image(SEGMENTATION_SIZE)

MODELS = [
    # Classification
    {"name": "resnet18", "family": "ResNet", "task": "Classification", "build": lambda: tvm.resnet18(), "input": imagenet},
    {"name": "resnet34", "family": "ResNet", "task": "Classification", "build": lambda: tvm.resnet34(), "input": imagenet},
    {"name": "resnet50", "family": "ResNet", "task": "Classification", "build": lambda: tvm.resnet50(), "input": imagenet},
    {"name": "resnet101", "family": "ResNet", "task": "Classification", "build": lambda: tvm.resnet101(), "input": imagenet},
    {"name": "vgg16", "family": "VGG", "task": "Classification", "build": lambda: tvm.vgg16(), "input": imagenet},
    {"name": "densenet121", "family": "DenseNet", "task": "Classification", "build": lambda: tvm.densenet121(), "input": imagenet},
    {"name": "inception_v3", "family": "Inception", "task": "Classification",
     "build": lambda: tvm.inception_v3(aux_logits=False, init_weights=True), "input": image(299)},
    {"name": "mobilenet_v3_small", "family": "MobileNetV3", "task": "Classification", "build": lambda: tvm.mobilenet_v3_small(), "input": imagenet},
    {"name": "mobilenet_v3_large", "family": "MobileNetV3", "task": "Classification", "build": lambda: tvm.mobilenet_v3_large(), "input": imagenet},
    {"name": "efficientnet_b0", "family": "EfficientNet", "task": "Classification", "build": lambda: tvm.efficientnet_b0(), "input": imagenet},
    {"name": "efficientnet_b2", "family": "EfficientNet", "task": "Classification", "build": lambda: tvm.efficientnet_b2(), "input": image(260)},
    {"name": "efficientnet_b4", "family": "EfficientNet", "task": "Classification", "build": lambda: tvm.efficientnet_b4(), "input": image(380)},
    {"name": "efficientnet_v2_s", "family": "EfficientNet", "task": "Classification", "build": lambda: tvm.efficientnet_v2_s(), "input": image(384)},
    {"name": "regnet_y_400mf", "family": "RegNet", "task": "Classification", "build": lambda: tvm.regnet_y_400mf(), "input": imagenet},
    {"name": "regnet_y_8gf", "family": "RegNet", "task": "Classification", "build": lambda: tvm.regnet_y_8gf(), "input": imagenet},
    {"name": "convnext_tiny", "family": "ConvNeXt", "task": "Classification", "build": lambda: tvm.convnext_tiny(), "input": imagenet},
    {"name": "convnext_small", "family": "ConvNeXt", "task": "Classification", "build": lambda: tvm.convnext_small(), "input": imagenet},
    {"name": "convnext_base", "family": "ConvNeXt", "task": "Classification", "build": lambda: tvm.convnext_base(), "input": imagenet},
    {"name": "vit_b_32", "family": "ViT", "task": "Classification", "build": lambda: tvm.vit_b_32(), "input": imagenet},
    {"name": "vit_b_16", "family": "ViT", "task": "Classification", "build": lambda: tvm.vit_b_16(), "input": imagenet},
    {"name": "vit_l_16", "family": "ViT", "task": "Classification", "build": lambda: tvm.vit_l_16(), "input": imagenet},
    {"name": "swin_t", "family": "Swin", "task": "Classification", "build": lambda: tvm.swin_t(), "input": imagenet},
    {"name": "swin_s", "family": "Swin", "task": "Classification", "build": lambda: tvm.swin_s(), "input": imagenet},
    {"name": "swin_b", "family": "Swin", "task": "Classification", "build": lambda: tvm.swin_b(), "input": imagenet},
    {"name": "maxvit_t", "family": "MaxViT", "task": "Classification", "build": lambda: tvm.maxvit_t(), "input": imagenet},

    # Self-supervised embeddings
    {"name": "dinov2_small", "family": "DINOv2", "task": "Image embedding", "build": dinov2(layers=12, hidden=384, heads=6), "input": imagenet},
    {"name": "dinov2_base", "family": "DINOv2", "task": "Image embedding", "build": dinov2(layers=12, hidden=768, heads=12), "input": imagenet},
    {"name": "dinov2_large", "family": "DINOv2", "task": "Image embedding", "build": dinov2(layers=24, hidden=1024, heads=16), "input": imagenet},

    # Detection
    {"name": "ssdlite320_mobilenet_v3", "family": "SSD", "task": "Detection",
     "build": lambda: Batched(tvd.ssdlite320_mobilenet_v3_large(weights_backbone=None)), "input": image(320)},
    {"name": "ssd300_vgg16", "family": "SSD", "task": "Detection",
     "build": lambda: Batched(tvd.ssd300_vgg16(weights_backbone=None)), "input": image(300)},
    {"name": "fasterrcnn_mobilenet_v3_320", "family": "Faster R-CNN", "task": "Detection",
     "build": lambda: Batched(tvd.fasterrcnn_mobilenet_v3_large_320_fpn(weights_backbone=None)), "input": detection},
    {"name": "fasterrcnn_mobilenet_v3", "family": "Faster R-CNN", "task": "Detection",
     "build": lambda: Batched(tvd.fasterrcnn_mobilenet_v3_large_fpn(weights_backbone=None)), "input": detection},
    {"name": "fasterrcnn_resnet50", "family": "Faster R-CNN", "task": "Detection",
     "build": lambda: Batched(tvd.fasterrcnn_resnet50_fpn(weights_backbone=None)), "input": detection},
    {"name": "retinanet_resnet50", "family": "RetinaNet", "task": "Detection",
     "build": lambda: Batched(tvd.retinanet_resnet50_fpn(weights_backbone=None)), "input": detection},
    {"name": "fcos_resnet50", "family": "FCOS", "task": "Detection",
     "build": lambda: Batched(tvd.fcos_resnet50_fpn(weights_backbone=None)), "input": detection},
    {"name": "detr_resnet50", "family": "DETR", "task": "Detection", "build": detr(), "input": detection},

    # Segmentation
    {"name": "unet_small", "family": "U-Net", "task": "Segmentation",
     "build": lambda: UNet(base=32, in_channels=GRAYSCALE, classes=2), "input": image(MEDICAL_SIZE, channels=GRAYSCALE)},
    {"name": "unet_base", "family": "U-Net", "task": "Segmentation",
     "build": lambda: UNet(base=64, in_channels=GRAYSCALE, classes=2), "input": image(MEDICAL_SIZE, channels=GRAYSCALE)},
    {"name": "lraspp_mobilenet_v3", "family": "LR-ASPP", "task": "Segmentation",
     "build": lambda: tvs.lraspp_mobilenet_v3_large(weights_backbone=None), "input": segmentation},
    {"name": "deeplabv3_mobilenet_v3", "family": "DeepLabV3", "task": "Segmentation",
     "build": lambda: tvs.deeplabv3_mobilenet_v3_large(weights_backbone=None), "input": segmentation},
    {"name": "deeplabv3_resnet50", "family": "DeepLabV3", "task": "Segmentation",
     "build": lambda: tvs.deeplabv3_resnet50(weights_backbone=None), "input": segmentation},
    {"name": "deeplabv3_resnet101", "family": "DeepLabV3", "task": "Segmentation",
     "build": lambda: tvs.deeplabv3_resnet101(weights_backbone=None), "input": segmentation},
    {"name": "maskrcnn_resnet50", "family": "Mask R-CNN", "task": "Segmentation",
     "build": lambda: Batched(tvd.maskrcnn_resnet50_fpn(weights_backbone=None)), "input": detection},
    {"name": "sam_vit_b_encoder", "family": "SAM", "task": "Segmentation",
     "build": lambda: SamVisionModel(SamVisionConfig()), "input": image(SAM_SIZE)},

    # Pose, depth, motion
    {"name": "keypointrcnn_resnet50", "family": "Keypoint R-CNN", "task": "Pose estimation",
     "build": lambda: Batched(tvd.keypointrcnn_resnet50_fpn(weights_backbone=None)), "input": detection},
    {"name": "dpt_base", "family": "DPT", "task": "Depth estimation",
     "build": lambda: DPTForDepthEstimation(DPTConfig()), "input": image(DPT_SIZE)},
    {"name": "raft_small", "family": "RAFT", "task": "Optical flow",
     "build": lambda: FramePair(tvf.raft_small()), "input": image(FLOW_HEIGHT, FLOW_WIDTH, channels=2 * RGB)},
    {"name": "raft_large", "family": "RAFT", "task": "Optical flow",
     "build": lambda: FramePair(tvf.raft_large()), "input": image(FLOW_HEIGHT, FLOW_WIDTH, channels=2 * RGB)},

    # Generation
    {"name": "dcgan_generator", "family": "DCGAN", "task": "Image generation",
     "build": lambda: DCGANGenerator(width=64), "input": features(GAN_NOISE, 1, 1)},
    {"name": "sd15_unet_step", "family": "Latent diffusion", "task": "Image generation",
     "build": sd15_unet(), "input": features(LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE)},
    {"name": "sd15_vae_decoder", "family": "Latent diffusion", "task": "Image generation",
     "build": sd15_vae_decoder(), "input": features(LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE)},
]
