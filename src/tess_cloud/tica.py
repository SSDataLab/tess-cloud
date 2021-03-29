"""Crawl the MIT TICA FFI data set.

Usage
-----
Use `save_tica_ffi_catalog(sector)` to create a catalog of TICA TESS FFI images.

"""
import asyncio
from urllib.error import HTTPError
from pathlib import Path
from typing import Union
from functools import lru_cache

from astropy.io.fits import Header
from astropy.time import Time
import pandas as pd
from pandas import DataFrame
import tqdm
import numpy as np

from .image import TessImage
from .imagelist import TessImageList
from . import DATADIR, log


TICA_MAST_PREFIX = "https://archive.stsci.edu/hlsps/tica/"


def list_tica_urls(sector=35) -> list:
    """Returns a list of TESS images produced by the TICA pipeline.

    Details: https://archive.stsci.edu/hlsp/tica
    """
    urls = []
    for orbit in [1, 2]:
        for camera in [1, 2, 3, 4]:
            for ccd in [1, 2, 3, 4]:
                urls += _list_tica_urls_by_ccd(
                    sector=sector, camera=camera, ccd=ccd, orbit=orbit
                )
    return urls


def _list_tica_urls_by_ccd(sector=35, camera=1, ccd=1, orbit=1) -> list:
    bundle_url = (
        f"https://archive.stsci.edu/hlsps/tica/bundles/s{sector:04d}/"
        f"hlsp_tica_tess_ffi_s{sector:04d}-o{orbit}-cam{camera}-ccd{ccd}_tess_v01_ffis.sh"
    )
    try:
        df = pd.read_fwf(bundle_url, colspecs=[(114, -1)], names=["url"])
        return df.url.tolist()
    except HTTPError:
        # HTTP 404 means the sector is not available yet in the archive
        return []


async def _get_tica_metadata_entry(url, sector=-1):
    img = TessImage(url)
    data_offset, hdrstr = await img._find_data_offset(return_header=True)
    hdr = Header.fromstring(hdrstr)
    return {
        "path": url.replace("https://archive.stsci.edu/hlsps/tica/", ""),
        "sector": sector,  # not in FITS header!
        "camera": hdr["CAMNUM"],
        "ccd": hdr["CCDNUM"],
        "start": Time(hdr["MJD-BEG"], format="mjd").iso[:19],
        "stop": Time(hdr["MJD-END"], format="mjd").iso[:19],
        "cadence": hdr["CADENCE"],
        "data_offset": data_offset,
    }


async def async_get_tica_metadata(sector=35):
    urls = list_tica_urls(sector)
    flist = [_get_tica_metadata_entry(url, sector=sector) for url in urls]
    tasks = [asyncio.create_task(f) for f in flist]
    for t in tqdm.tqdm(
        asyncio.as_completed(tasks), total=len(tasks), desc="Reading headers"
    ):
        await t
    return pd.DataFrame([t.result() for t in tasks])


def _tica_catalog_path(sector: int) -> Path:
    """Returns the filename of the FFI catalog of a given sector."""
    return DATADIR / Path(f"tess-s{sector:04d}-tica-ffi-catalog.parquet")


def save_tica_ffi_catalog(sector, path=None, overwrite=False) -> DataFrame:
    if path is None:
        path = _tica_catalog_path(sector=sector)
    if not overwrite and Path(path).exists():
        log.info(
            f"Skipping sector {sector}: file already exists ({path}).  Use `overwrite=True` to force-update."
        )
        return None
    df = asyncio.run(async_get_tica_metadata(sector=sector))
    log.info(f"Started writing {path}")
    df.to_parquet(path, compression="gzip")
    log.info(f"Finished writing {path}")
    return df


def list_tica_images(
    sector: int = 1,
    camera: int = None,
    ccd: int = None,
    time: Union[str, Time] = None,
) -> TessImageList:
    """Returns a list of TICA FFI images."""
    df = _load_tica_ffi_catalog(sector=sector)
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

    df["path"] = TICA_MAST_PREFIX + df["path"]

    # Add time column (TODO: move this to save_catalog)
    duration = Time(df.stop.iloc[0]) - Time(df.start.iloc[0])
    timeobj = Time(df.start.values.astype(str)) + (duration / 2)
    df["time"] = timeobj.iso

    # TODO: have this be part of save_catalog
    df["quality"] = np.zeros(len(df), dtype=int)
    df["cadenceno"] = df["cadence"]

    return TessImageList.from_catalog(df)


@lru_cache()
def _load_tica_ffi_catalog(sector: int) -> DataFrame:
    path = _tica_catalog_path(sector=sector)
    log.debug(f"Reading {path}")
    # TODO: move sort_values to `save_catalog`
    return pd.read_parquet(path).sort_values("path")
