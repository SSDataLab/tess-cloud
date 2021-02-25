# asyncio cannot be used in a Jupyter notebook environment
# without the following:
import nest_asyncio
nest_asyncio.apply()

import asyncio

from tess_locator import locate, TessCoordList
from tess_ephem import ephem

from .image import TessImage
from .targetpixelfile import TargetPixelFile
from .manifest import get_cloud_uri


def cutout_ffi(url, col, row, shape=(5, 5)) -> TargetPixelFile:
    """Retrieve a section from an FFI."""
    img = TessImage(url)
    cutout = img.cutout(col=col, row=row, shape=shape)
    return TargetPixelFile.from_cutouts([cutout]).to_lightkurve()


def cutout_header():
    # to do
    pass


def cutout(target: str, shape: tuple = (5, 5), sector: int = None, images: int = None, asynchronous=True) -> TargetPixelFile:
    """Returns a target pixel file."""
    crd = locate(target=target, sector=sector)[0]
    imagelist = crd.get_images()
    if images:
        imagelist = imagelist[:images]
    filenames = [img.filename for img in imagelist]
    uris = [get_cloud_uri(fn) for fn in filenames]

    if asynchronous:
        async def _get_cutouts():
            return await asyncio.gather(
                *[TessImage(uri).async_cutout(col=crd.column, row=crd.row, shape=shape)
                for uri in uris]
            )
        cutouts = asyncio.run(_get_cutouts())
    else:
        cutouts = [TessImage(uri).cutout(col=crd.column, row=crd.row, shape=shape)
                   for uri in uris]

    tpf = TargetPixelFile.from_cutouts(cutouts)
    return tpf.to_lightkurve()


def cutout_asteroid(target: str, shape: tuple = (10, 10), sector: int = None, images: int = None, asynchronous=True) -> TargetPixelFile:
    """Returns a moving Target Pixel File centered on an asteroid."""
    eph = ephem(target, verbose=True)
    if sector is None:
        all_sectors = eph.sector.unique()
        sector = all_sectors[0]
    #if len(all_sectors) > 0:
    #    raise ValueError(f"{target} has been observed in multiple sectors ({sectors}), please specify the sector using `sector=N`.")

    eph = eph[eph.sector == sector]
    if images:
        eph = eph[:images]

    crdlist = TessCoordList.from_pandas(eph)
    cutouts = []
    for crd in crdlist:
        ffi = crd.get_images()[0]  # there should only be one image because crdlist contains times
        uri = get_cloud_uri(ffi.filename)
        img = TessImage(uri)
        cutout = img.cutout(col=crd.column, row=crd.row, shape=shape)
        cutouts.append(cutout)
    tpf = TargetPixelFile.from_cutouts(cutouts)
    return tpf.to_lightkurve()
