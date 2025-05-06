


# Import mappings directly from the refactored nodes file
from .nodes.comfy_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# WEB_DIRECTORY might still be needed if there's a web component
WEB_DIRECTORY = "./web"

# Ensure __all__ exports the imported mappings and WEB_DIRECTORY
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']