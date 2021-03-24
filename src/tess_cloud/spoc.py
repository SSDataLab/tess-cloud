from functools import lru_cache

import pandas as pd

from .manifest import _load_ffi_manifest
from .image import TessImage, TessImageList


def list_spoc_images(
    sector: int = 1, camera: int = None, ccd: int = None, provider: str = "aws"
) -> TessImageList:
    """Returns a list of calibrated TESS FFI images.

    Defaults to AWS as provides because MAST HTTP yields lots of intermittent 503 errors.

    Parameters
    ----------
    provider : str
        "mast" or "aws".
    """
    if camera is None:
        camera = "\d"  # regex
    if ccd is None:
        ccd = "\d"  # regex
    if provider == "aws":
        return _list_spoc_images_aws(sector, camera, ccd)
    else:
        return _list_spoc_images_mast(sector, camera, ccd)


def _list_spoc_images_mast(sector, camera, ccd):
    df = _get_mast_bundle(sector=sector)
    mask = df.url.str.match(f".*tess(\d+)-s{sector:04d}-{camera}-{ccd}-\d+-._ffic.fits")
    return TessImageList([TessImage(url) for url in df[mask].url.values])


def _list_spoc_images_aws(sector, camera, ccd):
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
