from . import audio, image, structured, text, video

# Each module lists its models; the module name is the input type the report colors by.
MODELS = [
    {**model, "domain": module.__name__.rsplit(".", 1)[-1]}
    for module in (image, video, text, audio, structured)
    for model in module.MODELS
]
