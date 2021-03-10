import logging

__version__ = "0.1.0"

# Configure logging
log = logging.getLogger(__name__)
log.addHandler(logging.StreamHandler())

TESS_S3_BUCKET = "stpubdata"

from .manifest import list_images, get_uri
from .image import TessImage
from .cutout import cutout, cutout_ffi, cutout_asteroid

__all__ = [
    "list_images",
    "get_uri",
    "TessImage",
    "cutout",
    "cutout_ffi",
    "cutout_asteroid",
]
