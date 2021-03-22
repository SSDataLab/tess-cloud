import pandas as pd

from .image import TessImage, TessImageList


def list_spoc_images(sector=1, camera=1, ccd=1) -> TessImageList:
    """Returns a list of calibrated TESS FFI images."""
    bundle_url = f"https://archive.stsci.edu/missions/tess/download_scripts/sector/tesscurl_sector_{sector}_ffic.sh"
    df = pd.read_fwf(bundle_url, colspecs=[(61, -1)], names=["url"])
    mask = df.url.str.match(f".*tess(\d+)-s{sector:04d}-{camera}-{ccd}-\d+-._ffic.fits")
    return TessImageList([TessImage(url) for url in df[mask].url.values])
