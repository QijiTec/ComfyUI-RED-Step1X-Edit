from .nodes.step1x_edit_nodes import Step1XEditModelLoader, Step1XEditGenerateNode


NODE_CLASS_MAPPINGS = {
    "Step1XEditModelLoader": Step1XEditModelLoader,
    "Step1XEditGenerate": Step1XEditGenerateNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Step1XEditModelLoader": "Step1X-Edit Model Loader",
    "Step1XEditGenerate": "Step1X-Edit Generate"
}