from .nodes.step1x_edit_nodes import Step1XEditModelLoader, Step1XEditGenerateNode
from .nodes.step1x_edit_teacache_nodes import Step1XEditTeaCacheModelLoader, Step1XEditTeaCacheGenerateNode


NODE_CLASS_MAPPINGS = {
    "Step1XEditModelLoader": Step1XEditModelLoader,
    "Step1XEditGenerate": Step1XEditGenerateNode,
    "Step1XEditTeaCacheModelLoader": Step1XEditTeaCacheModelLoader,
    "Step1XEditTeaCacheGenerate": Step1XEditTeaCacheGenerateNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Step1XEditModelLoader": "Step1X-Edit Model Loader",
    "Step1XEditGenerate": "Step1X-Edit Generate",
    "Step1XEditTeaCacheModelLoader": "Step1X-Edit TeaCache Model Loader (2x faster)",
    "Step1XEditTeaCacheGenerate": "Step1X-Edit TeaCache Generate (2x faster)"
}