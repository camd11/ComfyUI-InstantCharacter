


# 注册节点
from .nodes.comfy_nodes import InstantCharacterLoader, InstantCharacterGenerate


NODE_CLASS_MAPPINGS = {
    "InstantCharacterLoader": InstantCharacterLoader,
    "InstantCharacterGenerate": InstantCharacterGenerate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "InstantCharacterLoader": "Load InstantCharacter Pipeline",
    "InstantCharacterGenerate": "Generate with InstantCharacter",
}
WEB_DIRECTORY = "./web"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', "WEB_DIRECTORY"]