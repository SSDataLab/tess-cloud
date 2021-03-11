import asyncio
import logging

__version__ = "0.1.0"

# Configure logging
log = logging.getLogger(__name__)
log.addHandler(logging.StreamHandler())

TESS_S3_BUCKET = "stpubdata"

# The maximum number of downloads to await at any given time is controlled using a semaphore.
# Too many parallel downloads may lead to read timeouts on slow connections.
MAX_CONCURRENT_DOWNLOADS = asyncio.Semaphore(100)

from .manifest import get_s3_uri
from .image import TessImage, TessImageList, list_images
from .cutout import cutout, cutout_ffi, cutout_asteroid

__all__ = [
    "list_images",
    "get_s3_uri",
    "TessImage",
    "TessImageList",
    "cutout",
    "cutout_ffi",
    "cutout_asteroid",
]
