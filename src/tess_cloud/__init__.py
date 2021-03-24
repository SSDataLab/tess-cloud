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

# Maximum number of images to cut out from at any given time;
# this enables the progress bar to progress smoothly.
MAX_CONCURRENT_CUTOUTS = asyncio.Semaphore(200)

from .manifest import get_s3_uri
from .image import TessImage, TessImageList
from .spoc import list_spoc_images
from .tica import list_tica_images
from .cutout import cutout, cutout_ffi, cutout_asteroid

__all__ = [
    "get_s3_uri",
    "TessImage",
    "TessImageList",
    "list_spoc_images",
    "list_tica_images",
    "cutout",
    "cutout_ffi",
    "cutout_asteroid",
]
