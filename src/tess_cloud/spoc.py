from functools import lru_cache
from urllib.error import HTTPError
from typing import Union

from astropy.time import Time
import pandas as pd
from pandas import DataFrame
import numpy as np

from .manifest import _load_ffi_manifest
from .image import TessImage
from .imagelist import TessImageList
from . import crawler, log


SPOC_AWS_PREFIX = "s3://stpubdata/tess/public/ffi"
SPOC_MAST_PREFIX = "https://archive.stsci.edu/missions/tess/ffi"


def list_spoc_images(
    sector: int = 1,
    camera: int = None,
    ccd: int = None,
    time: Union[str, Time] = None,
    provider: str = None,
) -> TessImageList:
    """Returns a list of calibrated TESS FFI images.

    Defaults to AWS as provides because MAST HTTP yields lots of intermittent 503 errors.

    Parameters
    ----------
    provider : str
        "mast" or "aws".
        Defaults to "aws".
    """
    df = _load_spoc_ffi_catalog(sector=sector)
    if camera:
        df = df[df.camera == camera]
    if ccd:
        df = df[df.ccd == ccd]
    if time:
        if isinstance(time, str):
            time = Time(time)
        begin = Time(np.array(df.start, dtype=str))
        end = Time(np.array(df.stop, dtype=str))
        mask = (time >= begin) & (time <= end)
        df = df[mask]

    if provider == "mast":
        df["path"] = SPOC_MAST_PREFIX + df["path"]
    else:
        df["path"] = SPOC_AWS_PREFIX + df["path"]

    return TessImageList.from_catalog(df)


@lru_cache()
def _load_spoc_ffi_catalog(sector: int) -> DataFrame:
    path = crawler._spoc_catalog_path(sector=sector)
    log.debug(f"Reading {path}")
    return pd.read_parquet(path)


###
# OLD FUNCTIONS
###


def _list_spoc_images_mast(sector, camera="\d", ccd="\d"):
    """
    Caution: this returns URLs of the form

       https://mast.stsci.edu/api/v0.1/Download/file/?uri=mast:TESS/product/tess2019086105934-s0010-1-1-0140-s_ffic.fits

    which are not suitable for tess-cloud because concurrent downloads from this sever yield a lot of HTTP 503 errors.
    We'll want to write an extra function to convert these URLs to the form:

       https://archive.stsci.edu/missions/tess/ffi/s0010/2019/086/1-1/tess2019086105934-s0010-1-1-0140-s_ffic.fits

    Which allows for many more concurrent requests.
    """
    # try:
    df = _get_mast_bundle(sector=sector)
    mask = df.url.str.match(f".*tess(\d+)-s{sector:04d}-{camera}-{ccd}-\d+-._ffic.fits")
    return TessImageList([TessImage(url) for url in df[mask].url.values])
    # except HTTPError:
    #    return TessImageList([])


def _list_spoc_images_aws(sector, camera="\d", ccd="\d"):
    """Returns a list of the FFIs for a given sector/camera/ccd."""
    ffi_files = _load_ffi_manifest()
    mask = ffi_files.path.str.match(
        f".*tess(\d+)-s{sector:04d}-{camera}-{ccd}-\d+-._ffic.fits"
    )
    return TessImageList(
        [TessImage("s3://stpubdata/" + x) for x in ffi_files[mask].path.values]
    )


@lru_cache()
def _get_mast_bundle(sector: int):
    bundle_url = f"https://archive.stsci.edu/missions/tess/download_scripts/sector/tesscurl_sector_{sector}_ffic.sh"
    return pd.read_fwf(bundle_url, colspecs=[(61, -1)], names=["url"])