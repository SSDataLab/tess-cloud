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


def cutout(target: str, sector: int = None, shape=(5, 5), images: int = None) -> TargetPixelFile:
    """Returns a target pixel file."""
    crd = locate(target=target, sector=sector)[0]
    imagelist = crd.get_images()
    if images:
        imagelist = imagelist[:images]
    filenames = [img.filename for img in imagelist]
    uris = [get_cloud_uri(fn) for fn in filenames]
    cutouts = []
    for idx, uri in enumerate(uris):
        img = TessImage(uri)
        cutout = img.cutout(col=crd.column, row=crd.row, shape=shape)
        cutouts.append(cutout)

    tpf = TargetPixelFile.from_cutouts(cutouts)
    return tpf.to_lightkurve()


def cutout_asteroid(target: str, shape=(10, 10), sector: int = None, images: int = None) -> TargetPixelFile:
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


async def _one_cutout(uri, column, row, shape):
    img = TessImage(uri)
    cutout = img.cutout(col=column, row=row, shape=shape)
    return cutout
