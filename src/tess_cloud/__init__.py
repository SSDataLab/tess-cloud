# asyncio cannot be used in a Jupyter notebook environment
# without first calling `nest_asyncio.apply()` following:
import nest_asyncio

nest_asyncio.apply()

import asyncio
import logging
from pathlib import Path

__version__ = "0.2.1"

# Configure logging
log = logging.getLogger(__name__)
log.addHandler(logging.StreamHandler())
log.setLevel("INFO")

# Where does this package store its embedded data?
PACKAGEDIR: Path = Path(__file__).parent.absolute()
DATADIR: Path = PACKAGEDIR / "data"

TESS_S3_BUCKET = "stpubdata"

# The maximum number of downloads to await at any given time is controlled using a semaphore.
# Too many parallel downloads may lead to read timeouts on slow connections.
MAX_CONCURRENT_DOWNLOADS = asyncio.Semaphore(200)

# Maximum number of images to cut out from at any given time;
# this enables the progress bar to progress smoothly.
MAX_CONCURRENT_CUTOUTS = asyncio.Semaphore(300)

from .manifest import get_s3_uri
from .image import TessImage
from .imagelist import TessImageList, list_images
from .cutout import cutout, cutout_ffi, cutout_asteroid

__all__ = [
    "get_s3_uri",
    "TessImage",
    "TessImageList",
    "list_images",
    "cutout",
    "cutout_ffi",
    "cutout_asteroid",
]
