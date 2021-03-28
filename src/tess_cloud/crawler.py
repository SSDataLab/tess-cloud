"""Crawl the TESS image data set.

Usage
-----
Use `save_spoc_ffi_catalog(sector)` to create a catalog of TESS FFI images.

"""
import asyncio
from pathlib import Path

from astropy.io.fits import Header
import s3fs
import pandas as pd
from pandas import DataFrame
import tqdm

from .image import TessImage
from . import DATADIR, log


def save_spoc_ffi_catalog(sector, path=None, overwrite=False) -> DataFrame:
    if path is None:
        path = _spoc_catalog_path(sector=sector)
    if not overwrite and Path(path).exists():
        log.info(
            f"Skipping sector {sector}: file already exists ({path}).  Use `overwrite=True` to force-update."
        )
        return None
    df = asyncio.run(async_get_spoc_metadata(sector=sector))
    log.info(f"Started writing {path}")
    df.to_parquet(path, compression="gzip")
    log.info(f"Finished writing {path}")
    return df


def _spoc_catalog_path(sector: int) -> Path:
    """Returns the filename of the FFI catalog of a given sector."""
    return DATADIR / Path(f"tess-s{sector:04d}-spoc-ffi-catalog.parquet")


async def async_get_spoc_metadata(sector=1):
    urls = list_spoc_urls(sector)
    flist = [_get_spoc_metadata_entry(url, sector=sector) for url in urls]
    tasks = [asyncio.create_task(f) for f in flist]
    for t in tqdm.tqdm(
        asyncio.as_completed(tasks), total=len(tasks), desc="Reading headers"
    ):
        await t
    return pd.DataFrame([t.result() for t in tasks])


def list_spoc_urls(sector=1, provider="aws"):
    fs = s3fs.S3FileSystem(anon=True)
    # urls = fs.glob(
    #    f"stpubdata/tess/public/ffi/s{sector:04d}/*/*/{camera}-{ccd}/**_ffic.fits"
    # )
    urls = fs.glob(f"stpubdata/tess/public/ffi/s{sector:04d}/**_ffic.fits")
    if provider == "mast":
        return [_aws_to_mast_url(u) for u in urls]
    return urls


def _aws_to_mast_url(url):
    """Convert a AWS S3 URI into a MAST URL.

    For example, the following S3 URI:
        s3://stpubdata/tess/public/ffi/s0020/2019/358/1-1/tess2019358235923-s0020-1-1-0165-s_ffic.fits
    will be translated into this MAST URL:
        https://archive.stsci.edu/missions/tess/ffi/s0020/2019/358/1-1/tess2019358235923-s0020-1-1-0165-s_ffic.fits
    """
    return url.replace(
        "stpubdata/tess/public", "https://archive.stsci.edu/missions/tess"
    )


async def _get_spoc_metadata_entry(url, sector=-1):
    img = TessImage(url)
    data_offset, hdrstr = await img._find_data_offset(return_header=True)
    hdr = Header.fromstring(hdrstr)
    return {
        "path": url.replace("stpubdata/tess/public/", ""),
        "sector": sector,  # not in FITS header!
        "camera": hdr["CAMERA"],
        "ccd": hdr["CCD"],
        "start": hdr["DATE-OBS"][:19],
        "stop": hdr["DATE-END"][:19],
        "quality": hdr["DQUALITY"],
        "data_offset": data_offset,
    }
