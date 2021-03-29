# TODO: set good guess for data_offset

# asyncio cannot be used in a Jupyter notebook environment
# without first calling `nest_asyncio.apply()` following:
from astropy.io.fits import column
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
from . import log


def cutout_ffi(url, column, row, shape=(5, 5)) -> TargetPixelFile:
    """Retrieve a section from an FFI."""
    img = TessImage(url)
    cutout = img.cutout(column=column, row=row, shape=shape)
    return TargetPixelFile.from_cutouts([cutout]).to_lightkurve()


def cutout_header():
    # to do
    pass


def cutout(
    target: str,
    shape: tuple = (5, 5),
    sector: int = None,
    author: str = "spoc",
    images: int = None,
) -> TargetPixelFile:
    """Returns a target pixel file."""
    locresult = locate(target=target, sector=sector)
    if len(locresult) < 1:
        raise ValueError("Target not observed by TESS yet.")
    if not sector:
        log.info(
            f"Target observed in {len(locresult)} sector(s): "
            f"{', '.join([str(r.sector) for r in locresult])}. \n"
            f"Using sector {locresult[0].sector}. "
            f"You can change this by passing the `sector` argument."
        )
    crd = locresult[0]
    imagelist = crd.list_images(author=author)
    if images:
        imagelist = imagelist[:images]

    return imagelist.cutout(column=crd.column, row=crd.row, shape=shape)


async def _get_cutouts(imagelist, crdlist, shape):
    async with aioboto3.client(
        "s3", config=Config(signature_version=UNSIGNED)
    ) as s3client:
        # Create list of functions to be executed
        flist = [
            img.async_cutout(
                column=crd.column, row=crd.row, shape=shape, client=s3client
            )
            for img, crd in zip(imagelist, crdlist)
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
    author: str = "spoc",
    images: int = None,
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
    imagelist = crdlist.list_images(author=author)
    filenames = [img.filename for img in imagelist]
    uris = [get_uri(fn) for fn in filenames]

    if asynchronous:
        cutouts = asyncio.run(_get_cutouts(uris, crdlist, shape))
    else:
        cutouts = [
            TessImage(uri).cutout(column=crd.column, row=crd.row, shape=shape)
            for uri, crd in zip(uris, crdlist)
        ]

    tpf = TargetPixelFile.from_cutouts(cutouts)
    return tpf.to_lightkurve()
