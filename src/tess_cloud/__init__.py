import logging

__version__ = "0.1.0"

# Configure logging
log = logging.getLogger(__name__)
log.addHandler(logging.StreamHandler())

TESS_S3_BUCKET = "stpubdata"

from .image import TessImage
from .cutout import cutout, cutout_ffi, cutout_asteroid

__all__ = [
    "TessImage",
    "cutout",
    "cutout_ffi",
    "cutout_asteroid"
]
