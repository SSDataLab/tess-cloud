# TODO: set good guess for data_offset

# asyncio cannot be used in a Jupyter notebook environment
# without the following:
import nest_asyncio

nest_asyncio.apply()

import asyncio
import warnings

import aioboto3
from botocore import UNSIGNED
from botocore.config import Config

import tqdm

from tess_locator import locate, TessCoordList
from tess_ephem import ephem

from .image import TessImage
from .targetpixelfile import TargetPixelFile
from .manifest import get_s3_uri as get_uri


def cutout_ffi(url, col, row, shape=(5, 5)) -> TargetPixelFile:
    """Retrieve a section from an FFI."""
    img = TessImage(url)
    cutout = img.cutout(col=col, row=row, shape=shape)
    return TargetPixelFile.from_cutouts([cutout]).to_lightkurve()


def cutout_header():
    # to do
    pass


def cutout(
    target: str,
    shape: tuple = (5, 5),
    sector: int = None,
    images: int = None,
    asynchronous=True,
) -> TargetPixelFile:
    """Returns a target pixel file."""
    crd = locate(target=target, sector=sector)[0]
    imagelist = crd.get_images()
    if images:
        imagelist = imagelist[:images]
    filenames = [img.filename for img in imagelist]
    uris = [get_uri(fn) for fn in filenames]
    crdlist = TessCoordList([crd]) * len(imagelist)

    if asynchronous:
        cutouts = asyncio.run(_get_cutouts(uris, crdlist, shape))
    else:
        cutouts = [
            TessImage(uri, data_offset=20160).cutout(
                col=crd.column, row=crd.row, shape=shape
            )
            for uri, crd in zip(uris, crdlist)
        ]

    tpf = TargetPixelFile.from_cutouts(cutouts)
    return tpf.to_lightkurve()


async def _get_cutouts(uris, crdlist, shape):
    async with aioboto3.client(
        "s3", config=Config(signature_version=UNSIGNED)
    ) as s3client:
        # Create list of functions to be executed
        flist = [
            TessImage(uri, data_offset=20160).async_cutout(
                col=crd.column, row=crd.row, shape=shape, client=s3client
            )
            for uri, crd in zip(uris, crdlist)
        ]
        # Create tasks for the sake of allowing a progress bar to be shown.
        # We'd want to use `asyncio.gather(*flist)` here to obtain the results in order,
        # but the progress par needs `asyncio.as_completed` to work.
        tasks = [asyncio.create_task(f) for f in flist]
        for t in tqdm.tqdm(
            asyncio.as_completed(tasks), total=len(tasks), desc="Downloading cutouts"
        ):
            await t
        # Now take care of getting the results in order
        results = [t.result() for t in tasks]
        return results


def cutout_asteroid(
    target: str,
    shape: tuple = (10, 10),
    sector: int = None,
    images: int = None,
    asynchronous=True,
) -> TargetPixelFile:
    """Returns a moving Target Pixel File centered on an asteroid."""
    eph = ephem(target, verbose=True)
    if sector is None:
        # Default to first available sector
        all_sectors = eph.sector.unique()
        sector = all_sectors[0]
        if len(all_sectors) > 0:
            warnings.warn(
                f"{target} has been observed in multiple sectors: {all_sectors}. "
                "Defaulting to `sector={sector}`."
                "You can change the sector by passing the `sector` keyword argument. "
            )

    eph = eph[eph.sector == sector]
    if images:
        eph = eph[:images]

    crdlist = TessCoordList.from_pandas(eph)
    imagelist = crdlist.get_images()
    filenames = [img.filename for img in imagelist]
    uris = [get_uri(fn) for fn in filenames]

    if asynchronous:
        cutouts = asyncio.run(_get_cutouts(uris, crdlist, shape))
    else:
        cutouts = [
            TessImage(uri, data_offset=20160).cutout(
                col=crd.column, row=crd.row, shape=shape
            )
            for uri, crd in zip(uris, crdlist)
        ]

    tpf = TargetPixelFile.from_cutouts(cutouts)
    return tpf.to_lightkurve()
