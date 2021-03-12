import pandas as pd

from .image import TessImage, TessImageList


def list_tica_images(sector=35, orbit=1, camera=1, ccd=1) -> TessImageList:
    bundle_url = f"https://archive.stsci.edu/hlsps/tica/bundles/s{sector:04d}/hlsp_tica_tess_ffi_s{sector:04d}-o{orbit}-cam{camera}-ccd{ccd}_tess_v01_ffis.sh"
    df = pd.read_fwf(bundle_url, header=0, names=("cmd", "flags", "path", "url"))
    return TessImageList([TessImage(url) for url in df.url.values])
