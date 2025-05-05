from .nodes.step1x_edit_nodes import REDStep1XEditModelLoader, REDStep1XEditGenerateNode


NODE_CLASS_MAPPINGS = {
    "REDStep1XEditModelLoader": REDStep1XEditModelLoader,
    "REDStep1XEditGenerate": REDStep1XEditGenerateNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Step1XEditModelLoader": "RED-Step1X-Edit Model Loader",
    "Step1XEditGenerate": "RED-Step1X-Edit Generate"
}