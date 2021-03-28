import pandas as pd
from urllib.error import HTTPError

from .image import TessImage
from .imagelist import TessImageList


def list_tica_images(sector=35, camera=1, ccd=1, orbit=None) -> TessImageList:
    """Returns a list of TESS images produced by the TICA pipeline.

    Details: https://archive.stsci.edu/hlsp/tica
    """
    if orbit:
        return _list_tica_images_by_orbit(sector, camera, ccd, orbit=orbit)

    # Else return both orbits
    l1 = _list_tica_images_by_orbit(sector, camera, ccd, orbit=1)
    l2 = _list_tica_images_by_orbit(sector, camera, ccd, orbit=2)
    return l1 + l2


def _list_tica_images_by_orbit(sector=35, camera=1, ccd=1, orbit=1) -> TessImageList:
    bundle_url = f"https://archive.stsci.edu/hlsps/tica/bundles/s{sector:04d}/hlsp_tica_tess_ffi_s{sector:04d}-o{orbit}-cam{camera}-ccd{ccd}_tess_v01_ffis.sh"
    try:
        df = pd.read_fwf(bundle_url, colspecs=[(114, -1)], names=["url"])
        return TessImageList([TessImage(url) for url in df.url.values])
    except HTTPError:
        # HTTP 404 means the sector is not available yet in the archive
        return TessImageList([])
